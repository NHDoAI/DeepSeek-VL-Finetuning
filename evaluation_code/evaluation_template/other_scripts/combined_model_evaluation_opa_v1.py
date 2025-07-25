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
import sys

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
    vl_chat_processor = VLChatProcessor.from_pretrained(base_model_name, trust_remote_code=True) # use the base model's tokenizer
    tokenizer = vl_chat_processor.tokenizer
    
    # Define and add new special tokens to the base tokenizer since it does not have them originally
    new_special_tokens = ["<LANE>", "<OBS>", "<DEC>"]
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

    if num_added_toks > 0:
        print(f"Added {num_added_toks} new special tokens to the tokenizer: {new_special_tokens}")
    else:
        print(f"The special tokens {new_special_tokens} were already present in the loaded tokenizer.")

    # Set pad_token if it's not already set (often eos_token is used as pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ('{tokenizer.eos_token}')")
        
    # Load configuration and update with tokenizer settings
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    config.pad_token_id = tokenizer.pad_token_id # Use the potentially updated pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id # Ensure BOS token is set if used


    
    # Load base model
    print(f"Loading base model '{base_model_name}' with 4-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        config=config, # Pass the updated config
        device_map="auto"
    )
    print("Base model loaded.")
    
    # Resize the base model's token embeddings to accommodate new tokens
    # This must be done *before* loading the PeftModel
    print(f"Resizing token embeddings of the language model component if necessary.")
    current_model_vocab_size = base_model.language_model.get_input_embeddings().weight.size(0)
    if current_model_vocab_size != len(tokenizer):
        print(f"Resizing base model token embeddings from {current_model_vocab_size} to {len(tokenizer)}.")
        base_model.language_model.resize_token_embeddings(len(tokenizer))
    else:
        print("Base model token embedding size already matches the tokenizer's vocabulary size.")
        
    # Load model with adapter
    print(f"Loading PEFT model with adapter from '{adapter_checkpoint}'...")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    print("PEFT model loaded.")
    
    # Merge adapter weights into base model
    print("Merging adapter weights into the base model...")
    merged_model = model_with_adapter.merge_and_unload()
    print("Adapter merged and unloaded.")
    
    fp16_model_path = os.path.join(output_dir, "fp16_model")

    # Save merged model
    print(f"Saving merged model to {fp16_model_path}...")
    merged_model.save_pretrained(fp16_model_path, safe_serialization=True)
    print("Merged model saved.")
    
    # Save processor and tokenizer (tokenizer now includes new tokens)
    print(f"Saving processor (with updated tokenizer) to {fp16_model_path}...")
    vl_chat_processor.save_pretrained(fp16_model_path)
    # tokenizer.save_pretrained(output_dir) # processor.save_pretrained should handle this
    print("Processor and tokenizer saved.")
    
    print(f"Merged fp16 model saved at {fp16_model_path}")
    return fp16_model_path

def quantize_model(model_path, output_dir):
    """Quantize the model to 4-bit."""
    print(f"Quantizing model from {model_path}...")
        # Prepare quantization config

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_path,            # the fp16 weights you just saved
        quantization_config=quantization_config,
        device_map="auto",          # loads sharded across available GPUs
    )

    merged_4bits_path = os.path.join(output_dir, "4bit_model")
    model_4bit.save_pretrained(merged_4bits_path, safe_serialization=True)
    vl_chat_processor.save_pretrained(merged_4bits_path)
    print(f"4 bits model quantized and saved at {merged_4bits_path}")
    return merged_4bits_path

# ============ MODEL INFERENCE FUNCTIONS ============

def run_inference(model_path, base_labels_dir, mode = "fp16"):
    """Run inference using prompts and image paths from JSON files."""
    print(f"Starting inference process...")
    print(f"Model path: {model_path}")
    print(f"Base labels directory: {base_labels_dir}")
    
    # Create main output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join("./eval_output", f"inference_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Load the chat processor and tokenizer from the model (once)
    print(f"Loading chat processor and tokenizer from {model_path}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = vl_chat_processor.tokenizer
    print("Chat processor and tokenizer loaded.")

    # Define and add new special tokens to the tokenizer (ensures consistency)
    new_special_tokens = ["<LANE>", "<OBS>", "<DEC>"]
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

    if num_added_toks > 0:
        print(f"Added {num_added_toks} new special tokens to the tokenizer for inference: {new_special_tokens}")
    else:
        print(f"The special tokens {new_special_tokens} were already present in the loaded tokenizer for inference.")

    # Set pad_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ('{tokenizer.eos_token}') for inference.")
        
    # Load the model (once)

    if mode == "fp16":
        print(f"Loading fp16 model from {model_path} for inference...")
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" # Automatically assigns device placement
        )
    elif mode == "4bit":
        print(f"Loading 4-bit model from {model_path} for inference...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    print("Model loaded for inference.")


    
    # Resize model's token embeddings if its vocabulary size doesn't match the tokenizer's.
    print("Checking model token embedding size for inference...")
    current_model_vocab_size = vl_gpt.language_model.get_input_embeddings().weight.size(0)
    if current_model_vocab_size != len(tokenizer):
        print(f"Resizing model token embeddings for inference from {current_model_vocab_size} to {len(tokenizer)}.")
        vl_gpt.language_model.resize_token_embeddings(len(tokenizer))
    else:
        print("Model token embedding size already matches the tokenizer's vocabulary size for inference.")

    vl_gpt = vl_gpt.eval()
    
    sub_dirs_to_process = ["real/labels/", "simulation/labels/"]

    for sub_dir_name in sub_dirs_to_process:
        current_labels_dir = os.path.join(base_labels_dir, sub_dir_name)
        current_output_sub_dir = os.path.join(main_output_dir, sub_dir_name)
        os.makedirs(current_output_sub_dir, exist_ok=True)

        print(f"\nProcessing JSON label files from: {current_labels_dir}")
        print(f"Saving results to: {current_output_sub_dir}")

        if not os.path.isdir(current_labels_dir):
            print(f"Warning: Directory not found {current_labels_dir}. Skipping.")
            continue
            
        json_label_files = [
            os.path.join(current_labels_dir, fname) for fname in os.listdir(current_labels_dir) 
            if fname.endswith('.json')
        ]

        if not json_label_files:
            print(f"No JSON label files found in {current_labels_dir}. Skipping.")
            continue
        
        print(f"Found {len(json_label_files)} JSON files to process in {sub_dir_name} directory")
        
        for i, label_file_path in enumerate(json_label_files):
            input_json_filename = os.path.basename(label_file_path)
            print(f"  Processing JSON file {i+1}/{len(json_label_files)} ({sub_dir_name}): {input_json_filename}")
            
            try:
                with open(label_file_path, "r") as f_json:
                    data = json.load(f_json)
                
                img_path_from_json = data["conversation"][0]["images"][0]
                current_user_prompt = data["conversation"][0]["content"]

            except Exception as e:
                print(f"    Error reading or parsing JSON file {input_json_filename}: {e}. Skipping.")
                continue

            img_filename = os.path.basename(img_path_from_json)
            base_name = os.path.splitext(img_filename)[0]
            
            conversation = [
                {
                    "role": "User",
                    "content": current_user_prompt,
                    "images": [img_path_from_json],
                },
                {"role": "Assistant", "content": ""}
            ]
            
            try:
                pil_images = load_pil_images(conversation)
            except Exception as e:
                print(f"    Error loading image {img_path_from_json} specified in {input_json_filename}: {e}. Skipping.")
                continue
            
            prepare_inputs = vl_chat_processor(
                conversations=conversation, 
                images=pil_images, 
                force_batchify=True
            ).to(vl_gpt.device)
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)
            
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            # Disable gradient calculations for inference
            with torch.no_grad():
                outputs = vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    max_new_tokens=65,
                    do_sample=False,
                    use_cache=True,
                )
            
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
            
            result = {
                "conversation": [
                    {
                        "role": "User",
                        "content": current_user_prompt,
                        "images": [img_path_from_json],
                    },
                    {
                        "role": "Assistant", 
                        "content": answer
                    }
                ]
            }
            
            inference_filename = f"{base_name}.json"
            result_path = os.path.join(current_output_sub_dir, inference_filename)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=4)
            
            print(f"    Processed image {img_filename} (from {input_json_filename}). Result saved to: {result_path}")
    
    print(f"\nAll results saved in: {main_output_dir}")
    return main_output_dir

# ============ CATEGORY EXTRACTION FUNCTIONS ============

def extract_categories_from_results(results_dir, output_dir, overall_timestamp):
    """Extract category information from JSON results, saving them into separate
    subdirectories for 'real' and 'simulation' data."""
    print(f"Starting category extraction process with timestamp: {overall_timestamp}...")
    os.makedirs(output_dir, exist_ok=True) # Ensure base output directory exists

    generated_csv_files = []

    # Define the types of data to process and their respective label subdirectories
    data_types_to_process = [
        {"name": "real", "label_subdir": "labels"},
        {"name": "simulation", "label_subdir": "labels"}
    ]

    for data_type_info in data_types_to_process:
        type_name = data_type_info["name"]
        label_subdir_name = data_type_info["label_subdir"]
        
        print(f"\nProcessing data type: '{type_name}'")

        current_input_labels_dir = os.path.join(results_dir, type_name, label_subdir_name)
        current_output_typed_dir = os.path.join(output_dir, type_name)
        os.makedirs(current_output_typed_dir, exist_ok=True)

        if not os.path.isdir(current_input_labels_dir):
            print(f"Warning: Input directory not found for '{type_name}': {current_input_labels_dir}. Skipping.")
            continue

        json_label_files = [
            os.path.join(current_input_labels_dir, fname)
            for fname in os.listdir(current_input_labels_dir)
            if fname.endswith('.json')
        ]

        if not json_label_files:
            print(f"No JSON label files found in {current_input_labels_dir} for '{type_name}'. Skipping.")
            # Optionally, create an empty CSV for this type
            empty_output_file = os.path.join(current_output_typed_dir, f'extraction_results_model_{type_name}_{overall_timestamp}_empty.csv')
            with open(empty_output_file, 'w', newline='') as f:
                fieldnames = ["file_name", "lane", "obstacle", "decision", "final_decision"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f"Empty results CSV saved for '{type_name}' at {empty_output_file}")
            generated_csv_files.append(empty_output_file)
            continue
        
        print(f"Found {len(json_label_files)} JSON files to process for '{type_name}' in {current_input_labels_dir}")
        
        data_text_current_type = []
        file_names_current_type = []
        
        for file_path in json_label_files:
            try:
                data = extract_text_from_json(file_path) # Assuming this function exists
                if len(data.get("conversation", [])) > 1 and "content" in data["conversation"][1]:
                    data_text_current_type.append(data["conversation"][1]["content"])
                    file_names_current_type.append(os.path.basename(file_path))
                else:
                    print(f"Warning: Could not find model response in {file_path} (type: {type_name}). Skipping.")
            except Exception as e:
                print(f"Warning: Error processing file {file_path} (type: {type_name}): {e}. Skipping.")

        if not data_text_current_type:
            print(f"No valid model responses found in JSON files for '{type_name}'.")
            continue

        detected_lanes = []
        detected_obstacles = []
        detected_decisions = []
        detected_final_decisions = []

        for data_item in data_text_current_type:
            # Lane detection
            lane_options = ["left lane", "right lane", "unclear"]
            best_score_for_lane_extraction = -1
            best_matched_phrase_for_lane = ""
            for option in lane_options:
                current_dynamic_query_lane = f"current lane is <LANE> {option}"
                temp_matched_phrase, temp_score = extract_phrase(data_item.lower(), current_dynamic_query_lane.lower())
                if temp_score > best_score_for_lane_extraction:
                    best_score_for_lane_extraction = temp_score
                    best_matched_phrase_for_lane = temp_matched_phrase
            detected_lane, _ = detect_category(best_matched_phrase_for_lane.lower(), lane_options, score_cutoff=50)
            detected_lanes.append(detected_lane)

            # Obstacle detection
            obstacles_options = ["far away", "near", "very close", "not on the same lane"]
            best_score_for_obstacle_extraction = -1
            best_matched_phrase_for_obstacle = ""
            for option in obstacles_options:
                current_dynamic_query_obstacle = f"Obstacles are <OBS> {option}"
                temp_matched_phrase, temp_score = extract_phrase(data_item.lower(), current_dynamic_query_obstacle.lower())
                if temp_score > best_score_for_obstacle_extraction:
                    best_score_for_obstacle_extraction = temp_score
                    best_matched_phrase_for_obstacle = temp_matched_phrase
            detected_obstacle, _ = detect_category(best_matched_phrase_for_obstacle.lower(), obstacles_options, score_cutoff=50)
            detected_obstacles.append(detected_obstacle)

            # Decision detection (from Scene Analysis)
            decision_options = ["straight forward", "slow cruise", "switch lane"]
            best_score_for_decision_extraction = -1
            best_matched_phrase_for_decision = ""
            for option in decision_options:
                current_dynamic_query_decision = f"optimal movement decision is <DEC> {option}"
                temp_matched_phrase, temp_score = extract_phrase(data_item.lower(), current_dynamic_query_decision.lower())
                if temp_score > best_score_for_decision_extraction:
                    best_score_for_decision_extraction = temp_score
                    best_matched_phrase_for_decision = temp_matched_phrase
            detected_decision, _ = detect_category(best_matched_phrase_for_decision.lower(), decision_options, score_cutoff=50)
            detected_decisions.append(detected_decision)

            # Final decision detection
            final_decision_options = ["straight forward", "slow cruise", "switch lane"]
            best_score_for_final_decision_extraction = -1
            best_matched_phrase_for_final_decision = ""
            for option in final_decision_options:
                current_dynamic_query_final_decision = f"[Final Decision]\n <DEC> {option}"
                temp_matched_phrase, temp_score = extract_phrase(data_item.lower(), current_dynamic_query_final_decision.lower())
                if temp_score > best_score_for_final_decision_extraction:
                    best_score_for_final_decision_extraction = temp_score
                    best_matched_phrase_for_final_decision = temp_matched_phrase
            
            temp_phrase_for_category_detection = ""
            if best_matched_phrase_for_final_decision:
                final_key_index = best_matched_phrase_for_final_decision.lower().rfind("[final decision]")
                if final_key_index != -1:
                    temp_phrase_for_category_detection = best_matched_phrase_for_final_decision[final_key_index:]
                else:
                    temp_phrase_for_category_detection = best_matched_phrase_for_final_decision
            else:
                temp_phrase_for_category_detection = best_matched_phrase_for_final_decision
            detected_final_decision, _ = detect_category(temp_phrase_for_category_detection.lower(), final_decision_options, score_cutoff=50)
            detected_final_decisions.append(detected_final_decision)

        results_current_type = []
        for i in range(len(file_names_current_type)):
            result = {
                "file_name": file_names_current_type[i],
                "lane": detected_lanes[i] if i < len(detected_lanes) else "N/a",
                "obstacle": detected_obstacles[i] if i < len(detected_obstacles) else "N/a",
                "decision": detected_decisions[i] if i < len(detected_decisions) else "N/a",
                "final_decision": detected_final_decisions[i] if i < len(detected_final_decisions) else "N/a"
            }
            results_current_type.append(result)
        
        if results_current_type:
            output_csv_filename = f'extraction_results_model_{type_name}_{overall_timestamp}.csv'
            output_csv_filepath = os.path.join(current_output_typed_dir, output_csv_filename)
            with open(output_csv_filepath, 'w', newline='') as f:
                fieldnames = ["file_name", "lane", "obstacle", "decision", "final_decision"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_current_type)
            print(f"Extraction results for '{type_name}' saved to {output_csv_filepath}")
            generated_csv_files.append(output_csv_filepath)
        else:
            print(f"No results to save for '{type_name}'.")

    if not generated_csv_files:
        print("No CSV files were generated in this run.")
    
    print(f"\nExtraction process complete. Generated CSVs: {generated_csv_files}")
    return generated_csv_files

# ============ COMPARISON FUNCTIONS ============

def compare_results_with_ground_truth(model_results_csv_paths_list, overall_timestamp):
    """
    Compares model-extracted categories with ground truth for multiple CSV files.
    The core comparison logic (iterating columns, calculating stats) remains the same.
    Outputs are saved to a 'categorical_comparison' directory, with subfolders for data types.
    """
    print(f"\nStarting comparison with ground truth using timestamp: {overall_timestamp}...")

    # Output directory for all comparisons, relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_comparison_dir = os.path.join(script_dir, "categorical_comparison")
    os.makedirs(base_comparison_dir, exist_ok=True)
    print(f"Comparison reports will be saved in subdirectories of: {base_comparison_dir}")

    generated_comparison_files = []

    for model_results_csv in model_results_csv_paths_list:
        model_file_parent_dir = os.path.dirname(model_results_csv)  # e.g., .../extraction_results/real
        data_type = os.path.basename(model_file_parent_dir)      # e.g., "real"

        if data_type not in ["real", "simulation"]: # Add other expected types if necessary
            print(f"  Warning: Could not determine data type for {model_results_csv} from dir '{data_type}'. Skipping.")
            continue

        ground_truth_csv = os.path.join(model_file_parent_dir, f"ground_truth_{data_type}.csv")

        if not os.path.exists(ground_truth_csv):
            print(f"  Warning: Ground truth file not found at {ground_truth_csv}. Skipping comparison for this file.")
            continue
        
        # Define the specific output directory for this data_type's comparison results
        type_specific_comparison_output_dir = os.path.join(base_comparison_dir, data_type)
        os.makedirs(type_specific_comparison_output_dir, exist_ok=True)

        print(f"\n  Processing comparison for data type: '{data_type}'")
        print(f"    Model results CSV: {model_results_csv}")
        print(f"    Ground truth CSV: {ground_truth_csv}")
        print(f"    Comparison output directory: {type_specific_comparison_output_dir}")

        try:
            df_ground_truth = pd.read_csv(ground_truth_csv)
            df_model = pd.read_csv(model_results_csv)
        except Exception as e:
            print(f"    Error reading CSV files for {data_type}: {e}. Skipping this comparison.")
            continue
        
        # Get ID column and other columns (EXISTING LOGIC)
        id_column = df_ground_truth.columns[0]
        columns_to_compare = [col for col in df_ground_truth.columns if col != id_column]
        
        # Define standard category orders (EXISTING LOGIC)
        category_orders = {
            'lane': ["left lane", "right lane", "unclear"],
            'obstacle': ["far away", "near", "very close", "not on the same lane"],
            'decision': ["straight forward", "slow cruise", "switch lane"],
            'final_decision': ["straight forward", "slow cruise", "switch lane"]
        }
        
        # Prepare results (EXISTING LOGIC)
        comparison_results_stats = {} # Renamed to avoid conflict if 'comparison_results' is used elsewhere
        category_counts_data = {}   # Renamed to avoid conflict
        
        for column in columns_to_compare:
            # Create merged dataframe for this column (EXISTING LOGIC)
            gt_col_name = f"{column}_ground_truth"
            model_col_name = f"{column}_model"
            
            # Ensure file_name column exists for merging, or use the determined id_column
            # The original code uses the first column as id_column.
            # If model CSVs also have 'file_name' as the first column, this is fine.
            # Otherwise, ensure consistent ID columns. For now, assume id_column is consistent.
            
            merged_df = pd.merge(
                df_ground_truth[[id_column, column]], 
                df_model[[id_column, column]],
                on=id_column,
                suffixes=('_ground_truth', '_model')
            )
            
            if merged_df.empty:
                print(f"    Warning: No common IDs ('{id_column}') found for column '{column}' between model and ground truth for {data_type}. Skipping this column.")
                continue

            # Add comparison column (EXISTING LOGIC)
            merged_df['is_same'] = merged_df[gt_col_name] == merged_df[model_col_name]
            
            # Calculate statistics (EXISTING LOGIC)
            total_rows = len(merged_df)
            matching_rows = merged_df['is_same'].sum()
            non_matching_rows = total_rows - matching_rows
            match_percentage = (matching_rows / total_rows * 100) if total_rows > 0 else 0
            
            # Store statistics (EXISTING LOGIC)
            comparison_results_stats[column] = {
                'total_rows': total_rows,
                'matching_rows': matching_rows,
                'non_matching_rows': non_matching_rows,
                'match_percentage': match_percentage
            }
            
            # Store category counts (EXISTING LOGIC)
            category_counts_data[column] = {
                'ground_truth': merged_df[gt_col_name].value_counts().to_dict(),
                'model': merged_df[model_col_name].value_counts().to_dict()
            }
            
            # Print statistics (EXISTING LOGIC - can be kept or removed if too verbose)
            print(f"\n    Statistics for '{column}' (data type: {data_type}):")
            print(f"    Total rows: {total_rows}")
            print(f"    Matching rows: {matching_rows} ({match_percentage:.2f}%)")
            print(f"    Non-matching rows: {non_matching_rows} ({100-match_percentage if total_rows > 0 else 0:.2f}%)")
            print(f"\n    Category Counts for '{column}' (data type: {data_type}):")
            print(f"    Ground Truth: {category_counts_data[column]['ground_truth']}")
            print(f"    Model: {category_counts_data[column]['model']}")
        
        # Save comparison results to CSV (ADAPTED OUTPUT PATH AND FILENAME)
        comparison_stats_file = os.path.join(type_specific_comparison_output_dir, f'comparison_stats_{data_type}_{overall_timestamp}.csv')
        
        with open(comparison_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Total Rows', 'Matching Rows', 'Non-Matching Rows', 'Match Percentage'])
            for category, stats in comparison_results_stats.items():
                writer.writerow([
                    category, 
                    stats['total_rows'], 
                    stats['matching_rows'], 
                    stats['non_matching_rows'], 
                    f"{stats['match_percentage']:.2f}%"
                ])
        generated_comparison_files.append(comparison_stats_file)
        
        # Save category counts to CSV (ADAPTED OUTPUT PATH AND FILENAME)
        category_counts_file = os.path.join(type_specific_comparison_output_dir, f'category_counts_{data_type}_{overall_timestamp}.csv')
        
        with open(category_counts_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Type', 'Value', 'Count'])
            
            for category_key in columns_to_compare: # Iterate through the order they were processed
                if category_key in category_orders and category_key in category_counts_data:
                    ordered_values = category_orders[category_key]
                    
                    # Ground truth counts for this category
                    ground_truth_counts = category_counts_data[category_key]['ground_truth']
                    for value in ordered_values:
                        count = ground_truth_counts.get(value, 0)
                        writer.writerow([category_key, 'ground_truth', value, count])
                    # Add any ground truth values not in ordered_values (optional, for completeness)
                    for value, count in ground_truth_counts.items():
                        if value not in ordered_values:
                             writer.writerow([category_key, 'ground_truth', value, count])

                    # Model counts for this category
                    model_counts = category_counts_data[category_key]['model']
                    for value in ordered_values:
                        count = model_counts.get(value, 0)
                        writer.writerow([category_key, 'model', value, count])
                    # Add any model values not in ordered_values (optional, for completeness)
                    for value, count in model_counts.items():
                        if value not in ordered_values:
                            writer.writerow([category_key, 'model', value, count])
                elif category_key in category_counts_data: # Fallback if category not in category_orders
                     print(f"    Warning: Category '{category_key}' not found in category_orders. Writing counts without specific order.")
                     ground_truth_counts = category_counts_data[category_key]['ground_truth']
                     for value, count in ground_truth_counts.items():
                        writer.writerow([category_key, 'ground_truth', value, count])
                     model_counts = category_counts_data[category_key]['model']
                     for value, count in model_counts.items():
                        writer.writerow([category_key, 'model', value, count])


        generated_comparison_files.append(category_counts_file)
        print(f"    Comparison statistics for {data_type} saved to {comparison_stats_file}")
        print(f"    Category counts for {data_type} saved to {category_counts_file}")

    if not generated_comparison_files:
        print("No comparison CSV files were generated.")
    else:
        print(f"\nComparison process complete. Generated files: {generated_comparison_files}")
    return generated_comparison_files

# ============ MAIN FUNCTION ============

def main():
    """Main function to run the entire pipeline."""
    
    # --- Setup logging to file ---
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"evaluation_pipeline_log_{log_timestamp}.log"
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    with open(log_file_name, 'w') as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        
        try:
            print(f"Logging to file: {os.path.abspath(log_file_name)}\n")
            print("Starting model evaluation pipeline...\n")
            
            # Define paths and directories
            base_model_name = os.getenv('MODEL_1.3B_CHKPOINT')
            adapter_checkpoint = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/1.3b_Option-A_version2_fixed/b6-s322_full-epoch_loss/first_epoch_chkpoint/adapter/"
            merged_model_dir = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/1.3b_Option-A_version2_fixed/b6-s322_full-epoch_loss/first_epoch_chkpoint/merged_model_first_epoch_chkpoint/4bit_model"
            test_labels_dir = "./test/"
            extraction_dir = "./extraction_results/"
            #ground_truth_csv = "./extraction_results/extraction_results_ground-truth.csv"
            
            merge_model_flag = True
            quantize_model_flag = True

            if merge_model_flag:
                # Step 1: Merge model with adapter
                print("\n===== STEP 1: MERGING MODEL WITH ADAPTER =====")
                merged_fp16_model_path = merge_model(base_model_name, adapter_checkpoint, merged_model_dir)
                if quantize_model_flag:
                    merged_4bit_model_path = quantize_model(merged_fp16_model_path, merged_model_dir)
                    merged_model_path = merged_4bit_model_path
                else:
                    merged_model_path = merged_fp16_model_path
            else:
                merged_model_path = merged_model_dir # manually set the path to the merged model
            
            # Step 2: Run inference on test images

            run_inference_flag = True
            if run_inference_flag:
                print("\n===== STEP 2: RUNNING INFERENCE ON TEST IMAGES =====")
                inference_results_dir = run_inference(merged_model_path, test_labels_dir, mode="4bit")

            else:
                inference_results_dir = "./eval_output/inference_20250525_000824"

            # Step 3: Extract categories from results

            extract_cat_flag = True
            if extract_cat_flag:
                
                print("\n===== STEP 3: EXTRACTING CATEGORIES FROM RESULTS =====")
                overall_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_results_csv = extract_categories_from_results(inference_results_dir, extraction_dir, overall_timestamp)
            else:
                model_results_csv = ["./extraction_results/real/extraction_results_model_real_20250525_010633.csv", "./extraction_results/simulation/extraction_results_model_simulation_20250525_010633.csv"]
                overall_timestamp = "20250525_010633"

            # Step 4: Compare results with ground truth
            print("\n===== STEP 4: COMPARING RESULTS WITH GROUND TRUTH =====")
            # comparison_stats_file, category_counts_file = compare_results_with_ground_truth(
            #     model_results_csv, 
            #     ground_truth_csv, 
            #     extraction_dir
            # )
            compare_results_with_ground_truth(model_results_csv, overall_timestamp)

            # print("\nEvaluation pipeline completed successfully!")
            # print(f"Merged model: {merged_model_path}")
            # print(f"Inference results: {inference_results_dir}")
            # print(f"Model results CSV: {model_results_csv}")
            # print(f"\nEvaluation run {overall_timestamp} complete. All outputs in: {extraction_dir}")
            # print(f"Comparison reports (comparison_stats and category_counts) are in: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'categorical_comparison')}")
            
            # The following print statements will now go to the log file
            if 'merged_model_path' in locals(): # Check if defined
                print(f"Final merged model path used: {merged_model_path}")
            if 'inference_results_dir' in locals():
                print(f"Final inference results directory: {inference_results_dir}")
            if 'model_results_csv' in locals():
                print(f"Final model results CSV paths: {model_results_csv}")
            if 'overall_timestamp' in locals():
                 print(f"\nEvaluation run {overall_timestamp} processing complete.")
                 if 'extraction_dir' in locals():
                    print(f"Extraction outputs are in: {extraction_dir}")
                 print(f"Comparison reports (if generated) are in: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'categorical_comparison')}")

            print("\nEvaluation pipeline script finished execution.")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # The 'with open' statement ensures log_file is closed automatically

    # This message will print to the actual console after redirection is reverted
    print(f"All console output from the script run has been logged to: {os.path.abspath(log_file_name)}")

    # if not model_results_csv:
    #     print("No model category CSVs were generated by extract_categories_from_results. Halting comparison.")
    # else:
    #     # The compare_results_with_ground_truth function will now create 'categorical_comparison' directory
    #     # and its subdirectories ('real', 'simulation') for its outputs.
    #     # It takes the list of model CSVs and the overall_timestamp.
        
if __name__ == "__main__":
    main() 