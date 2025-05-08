"""
Combined script for model merging, evaluation, and analysis.
This script:
1. Merges adapter weights into a base model
2. Runs inference on test images
3. Extracts category information from results
4. Compares results with ground truth
"""

import os
import glob
import torch
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from rapidfuzz import fuzz
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoConfig
)
from peft import PeftModel, PeftConfig
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# Load environment variables
load_dotenv()

# ============ HELPER FUNCTIONS FOR TEXT EXTRACTION ============

def extract_phrase(text_target, phrase_query, score_cutoff=0):
    """Extract a phrase from target text using fuzzy matching."""
    split_target = text_target.split()
    split_query = phrase_query.split()
    query_word_count = len(split_query)
    target_word_count = len(split_target)
    best_score = 0
    best_match = ""
    for i in range(target_word_count + 1 - query_word_count):
        candidate = " ".join(split_target[i:i+query_word_count])
        score = fuzz.ratio(candidate, phrase_query, score_cutoff=score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            best_match = candidate
    return best_match, best_score

def detect_category(text_target, categories_query, score_cutoff=75):
    """Detect category from text using fuzzy matching."""
    best_score = 0
    detected_category = "N/a"
    for category in categories_query:
        score = fuzz.partial_ratio(text_target, category, score_cutoff=score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            detected_category = category
    return detected_category, best_score

def extract_text_from_json(json_file):
    """Extract text content from JSON file."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# ============ MODEL MERGING FUNCTIONS ============

def merge_model(base_model_name, adapter_checkpoint, output_dir):
    """Merge adapter weights into base model and save the result."""
    print(f"Starting model merging process...")
    print(f"Base model: {base_model_name}")
    print(f"Adapter checkpoint: {adapter_checkpoint}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor and tokenizer
    vl_chat_processor = VLChatProcessor.from_pretrained(base_model_name)
    tokenizer = vl_chat_processor.tokenizer
    
    # Prepare quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load configuration
    config = AutoConfig.from_pretrained(base_model_name)
    config.pad_token_id = tokenizer.eos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        config=config,
        device_map="auto"
    )
    
    # Load model with adapter
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    
    # Merge adapter weights into base model
    merged_model = model_with_adapter.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_dir)
    
    # Save processor and tokenizer
    vl_chat_processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Merged model saved at {output_dir}")
    return output_dir

# ============ MODEL INFERENCE FUNCTIONS ============

def run_inference(model_path, images_dir):
    """Run inference on images and save results."""
    print(f"Starting inference process...")
    print(f"Model path: {model_path}")
    print(f"Images directory: {images_dir}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./eval_output", f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the chat processor and tokenizer from the model
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    # Load the model
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    vl_gpt = vl_gpt.eval()
    
    # Define user prompt
    user_prompt = """<image_placeholder> Analyze the given image from the vehicle's front-facing camera. First determine if the vehicle is on the left or the right lane. Then determine if there is an obstacle directly ahead in the lane the vehicle is currently occupying or not. If an obstacle is present, determine how far away the obstacle is and decide the best movement option to keep going forward or switch to the other lane or navigate safely.
    Guidelines for Decision Making:
    1) First determine if the vehicle is currently on the right lane or left lane of the road.
    2) Then determine if no obstacle is detected in the current lane or if it is far away, then the vehicle should continue moving "straight forward".
    3) If an obstacle is detected in the current lane (not the other lane) and it is near to the vehicle, it should "slow cruise" cautiously until the vehicle is very close to the obstacle.
    4) If an obstacle is detected on the current lane and it is very close to the vehicle (there is little to no road left between the vehicle and the obstacle), the vehicle should switch to the other lane by either "change lane left" or "change lane right", following this logic:
    - If the vehicle is currently on the left lane, it should "change lane right".
    - If the vehicle is currently on the right lane, it should "change lane left".
Response Format (Strict Adherence Required):
[Scene analysis and Reasoning]
- The vehicle is currently in the {left/right} lane.
- Obstacles are {choose one from: "not on the same lane", "far away", "near", "very close"}.
- Based on these conditions, the optimal movement decision is: {decision from: "straight forward", "slow cruise", "change lane left", "change lane right"}.
[Final Decision]
{decision}"""
    
    # Get all image files in the folder
    test_files = [
        os.path.join(images_dir, fname) for fname in os.listdir(images_dir) 
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')
    ]
    
    print(f"Found {len(test_files)} images to process")
    
    # Process each image
    for i, img_path in enumerate(test_files):
        print(f"Processing image {i+1}/{len(test_files)}: {os.path.basename(img_path)}")
        
        # Extract image filename
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        
        # Create conversation with user prompt and image
        conversation = [
            {
                "role": "User",
                "content": user_prompt,
                "images": [img_path],
            },
            {"role": "Assistant", "content": ""}
        ]
        
        # Load image
        pil_images = load_pil_images(conversation)
        
        # Prepare inputs for the model
        prepare_inputs = vl_chat_processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(vl_gpt.device)
        prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)
        
        # Run image encoder to get image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        
        # Decode the answer
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Save result
        result = {
            "conversation": [
                {
                    "role": "User",
                    "content": user_prompt,
                    "images": [img_path],
                },
                {
                    "role": "Assistant", 
                    "content": answer
                }
            ]
        }
        
        # Save as JSON
        inference_filename = f"{base_name}.json"
        result_path = os.path.join(output_dir, inference_filename)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        
        print(f"  Result saved to: {result_path}")
    
    print(f"All results saved in: {output_dir}")
    return output_dir

# ============ CATEGORY EXTRACTION FUNCTIONS ============

def extract_categories_from_results(results_dir, output_dir):
    """Extract category information from JSON results."""
    print(f"Extracting categories from results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the results directory
    test_files = [
        os.path.join(results_dir, fname) 
        for fname in os.listdir(results_dir) 
        if fname.endswith('.json')
    ]
    
    data_text = []
    file_names = []
    
    for file in test_files:
        data = extract_text_from_json(file)
        data_text.append(data["conversation"][1]["content"])
        file_names.append(os.path.basename(file))
    
    detected_lanes = []
    detected_obstacles = []
    detected_decisions = []
    detected_final_decisions = []
    
    for data in data_text:
        # Lane detection
        current_query = "The vehicle is currently in the rl lane"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        lane_options = ["left lane", "right lane"]
        detected_lane, score = detect_category(matched_phrase.lower(), lane_options, score_cutoff=76)
        detected_lanes.append(detected_lane)
        
        # Obstacle detection
        current_query = "Obstacles are not on the same lane"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        obstacles_options = ["far away", "near", "very close", "not on the same lane"]
        detected_obstacle, score = detect_category(matched_phrase.lower(), obstacles_options, score_cutoff=76)
        detected_obstacles.append(detected_obstacle)
        
        # Decision detection
        current_query = "optimal movement decision is: change lane left"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        decision_options = ["straight forward", "slow cruise", "change lane left", "change lane right"]
        detected_decision, score = detect_category(matched_phrase.lower(), decision_options, score_cutoff=76)
        detected_decisions.append(detected_decision)
        
        # Final decision detection
        current_query = "[Final Decision]\nchange lane left"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        
        final_key_index = matched_phrase.rfind("[final")
        if final_key_index != -1:
            matched_phrase = matched_phrase[final_key_index:]
            
        final_decision_options = ["straight forward", "slow cruise", "change lane left", "change lane right"]
        detected_final_decision, score = detect_category(matched_phrase.lower(), final_decision_options, score_cutoff=76)
        detected_final_decisions.append(detected_final_decision)
    
    # Create dictionary to store extracted information
    results = []
    for i in range(len(file_names)):
        result = {
            "file_name": file_names[i],
            "lane": detected_lanes[i] if i < len(detected_lanes) else "N/a",
            "obstacle": detected_obstacles[i] if i < len(detected_obstacles) else "N/a",
            "decision": detected_decisions[i] if i < len(detected_decisions) else "N/a",
            "final_decision": detected_final_decisions[i] if i < len(detected_final_decisions) else "N/a"
        }
        results.append(result)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'extraction_results_model_{timestamp}.csv')
    with open(output_file, 'w', newline='') as f:
        fieldnames = ["file_name", "lane", "obstacle", "decision", "final_decision"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Extraction complete. Results saved to {output_file}")
    return output_file

# ============ COMPARISON FUNCTIONS ============

def compare_results_with_ground_truth(model_results_csv, ground_truth_csv, output_dir):
    """Compare model results with ground truth and calculate statistics."""
    print(f"Comparing results with ground truth...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSVs
    df_ground_truth = pd.read_csv(ground_truth_csv)
    df_model = pd.read_csv(model_results_csv)
    
    # Get ID column and other columns
    id_column = df_ground_truth.columns[0]
    columns_to_compare = [col for col in df_ground_truth.columns if col != id_column]
    
    # Define standard category orders
    category_orders = {
        'lane': ["left lane", "right lane"],
        'obstacle': ["far away", "near", "very close", "not on the same lane"],
        'decision': ["straight forward", "slow cruise", "change lane right", "change lane left"],
        'final_decision': ["straight forward", "slow cruise", "change lane right", "change lane left"]
    }
    
    # Prepare results
    comparison_results = {}
    category_counts = {}
    
    for column in columns_to_compare:
        # Create merged dataframe for this column
        gt_col_name = f"{column}_ground_truth"
        model_col_name = f"{column}_model"
        
        merged_df = pd.merge(
            df_ground_truth[[id_column, column]], 
            df_model[[id_column, column]],
            on=id_column,
            suffixes=('_ground_truth', '_model')
        )
        
        # Add comparison column
        merged_df['is_same'] = merged_df[gt_col_name] == merged_df[model_col_name]
        
        # Calculate statistics
        total_rows = len(merged_df)
        matching_rows = merged_df['is_same'].sum()
        non_matching_rows = total_rows - matching_rows
        match_percentage = (matching_rows / total_rows * 100) if total_rows > 0 else 0
        
        # Store statistics
        comparison_results[column] = {
            'total_rows': total_rows,
            'matching_rows': matching_rows,
            'non_matching_rows': non_matching_rows,
            'match_percentage': match_percentage
        }
        
        # Store category counts
        category_counts[column] = {
            'ground_truth': merged_df[gt_col_name].value_counts().to_dict(),
            'model': merged_df[model_col_name].value_counts().to_dict()
        }
        
        # Print statistics
        print(f"\nStatistics for '{column}':")
        print(f"Total rows: {total_rows}")
        print(f"Matching rows: {matching_rows} ({match_percentage:.2f}%)")
        print(f"Non-matching rows: {non_matching_rows} ({100-match_percentage:.2f}%)")
        print("\nCategory Counts:")
        print(f"Ground Truth: {category_counts[column]['ground_truth']}")
        print(f"Model: {category_counts[column]['model']}")
    
    # Save comparison results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_stats_file = os.path.join(output_dir, f'comparison_stats_{timestamp}.csv')
    
    with open(comparison_stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Total Rows', 'Matching Rows', 'Non-Matching Rows', 'Match Percentage'])
        for category, stats in comparison_results.items():
            writer.writerow([
                category, 
                stats['total_rows'], 
                stats['matching_rows'], 
                stats['non_matching_rows'], 
                f"{stats['match_percentage']:.2f}%"
            ])
    
    # Save category counts to CSV using the new organized approach
    category_counts_file = os.path.join(output_dir, f'category_counts_{timestamp}.csv')
    
    with open(category_counts_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Type', 'Value', 'Count'])
        
        # First write all ground truth categories in order
        for category in columns_to_compare:
            ordered_values = category_orders[category]
            ground_truth_counts = category_counts[category]['ground_truth']
            
            # Write values in the predefined order
            for value in ordered_values:
                count = ground_truth_counts.get(value, 0)
                writer.writerow([category, 'ground_truth', value, count])
        
        # Then write all model categories in order
        for category in columns_to_compare:
            ordered_values = category_orders[category]
            model_counts = category_counts[category]['model']
            
            # Write values in the predefined order
            for value in ordered_values:
                count = model_counts.get(value, 0)
                writer.writerow([category, 'model', value, count])
    
    print(f"Comparison statistics saved to {comparison_stats_file}")
    print(f"Category counts saved to {category_counts_file}")
    
    return comparison_stats_file, category_counts_file

# ============ MAIN FUNCTION ============

def main():
    """Main function to run the entire pipeline."""
    print("Starting model evaluation pipeline...\n")
    
    # Define paths and directories
    base_model_name = os.getenv('MODEL_1.3B_CHKPOINT')
    adapter_checkpoint = "/home/ubuntu/DeepSeek-VL-Finetuning/training_code/cluster_training/after_first-epoch_fix_models/1.3B_lr-scheduler_no-train-shuffle_smallset/best_epoch/adapter/"
    merged_model_dir = "/home/ubuntu/DeepSeek-VL-Finetuning/training_code/cluster_training/after_first-epoch_fix_models/1.3B_lr-scheduler_no-train-shuffle_smallset/best_epoch/merged_model_first_epoch"
    test_images_dir = "./test/images"
    extraction_dir = "./extraction_results"
    ground_truth_csv = "./extraction_results/extraction_results_ground-truth.csv"
    
    merge_model_flag = False

    if merge_model_flag:
        # Step 1: Merge model with adapter
        print("\n===== STEP 1: MERGING MODEL WITH ADAPTER =====")
        merged_model_path = merge_model(base_model_name, adapter_checkpoint, merged_model_dir)
    else:
        merged_model_path = merged_model_dir
    
    # Step 2: Run inference on test images

    run_inference_flag = False
    if run_inference_flag:
        print("\n===== STEP 2: RUNNING INFERENCE ON TEST IMAGES =====")
        inference_results_dir = run_inference(merged_model_path, test_images_dir)
    else:
        inference_results_dir = "./eval_output/inference_20250430_230421"

    
    # Step 3: Extract categories from results
    print("\n===== STEP 3: EXTRACTING CATEGORIES FROM RESULTS =====")
    model_results_csv = extract_categories_from_results(inference_results_dir, extraction_dir)
    
    # Step 4: Compare results with ground truth
    print("\n===== STEP 4: COMPARING RESULTS WITH GROUND TRUTH =====")
    comparison_stats_file, category_counts_file = compare_results_with_ground_truth(
        model_results_csv, 
        ground_truth_csv, 
        extraction_dir
    )
    
    print("\nEvaluation pipeline completed successfully!")
    print(f"Merged model: {merged_model_path}")
    print(f"Inference results: {inference_results_dir}")
    print(f"Model results CSV: {model_results_csv}")
    print(f"Comparison statistics: {comparison_stats_file}")
    print(f"Category counts: {category_counts_file}")

if __name__ == "__main__":
    main() 