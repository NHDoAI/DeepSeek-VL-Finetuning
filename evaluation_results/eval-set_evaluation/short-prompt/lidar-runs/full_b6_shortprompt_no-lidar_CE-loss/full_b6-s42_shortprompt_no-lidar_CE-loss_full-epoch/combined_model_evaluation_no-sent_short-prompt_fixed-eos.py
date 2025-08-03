"""
Combined script for model merging, evaluation, and analysis.
This script:
1. Merges adapter weights into a base model
2. Runs inference on test images
3. Extracts category information from results
4. Compares results with ground truth

Note: saves results directly to categorical_comparison folder.
"""

import os
import glob
import torch
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from rapidfuzz import fuzz
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoConfig,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
import sys
from sklearn.metrics import confusion_matrix, fbeta_score, cohen_kappa_score
import io

# Load environment variables
load_dotenv()

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)  # Set working directory to script location

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
    
    # Load model with adapter
    print(f"Loading PEFT model with adapter from '{adapter_checkpoint}'...")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    print("PEFT model loaded.")
    
    # Merge adapter weights into base model
    print("Merging adapter weights into the base model...")
    merged_model = model_with_adapter.merge_and_unload()
    print("Adapter merged and unloaded.")

        # --- THE FIX: Ensure generation_config exists before modifying it ---
    print("Updating merged model's generation config with correct special token IDs...")
    
    # Check if the generation_config attribute is None
    if merged_model.generation_config is None:
        print("INFO: merged_model.generation_config was None. Initializing a new one from the main model config.")
        # If it's None, create a new GenerationConfig from the model's main config
        merged_model.generation_config = GenerationConfig.from_model_config(merged_model.config)

    # Now that we are GUARANTEED that merged_model.generation_config is an object,
    # we can safely modify its attributes.
    print("Updating merged model's generation config with correct special token IDs...")
    merged_model.generation_config.pad_token_id = tokenizer.pad_token_id
    merged_model.generation_config.eos_token_id = tokenizer.eos_token_id
    # ---------------------------------------------------

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
                    max_new_tokens=20,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id # added eos_token_id to config of generate method. This is fix to bug where generate method did not recognize eos_token_id from tokenizer.
                )
            

            # Now, you need to decode correctly, stopping at the assistant's part
            # The output contains both the input prompt tokens and the new tokens.
            # You only want to decode the generated part.
            # input_token_len = prepare_inputs.input_ids.shape[1]
            # generated_tokens = outputs[0][input_token_len:]

            # # 2. Move the new tokens to the CPU, convert to a list, and then decode
            # answer = tokenizer.decode(generated_tokens.cpu().tolist(), skip_special_tokens=True)
            generated_tokens = outputs[0]
            
            # Now, decode these generated tokens.
            answer = tokenizer.decode(generated_tokens.cpu().tolist(), skip_special_tokens=True)
            
            result = {
                "conversation": [
                    {
                        "role": "User",
                        "content": current_user_prompt,
                        "images": [img_path_from_json],
                    },
                    {
                        "role": "Assistant", 
                        "content": answer.strip()
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
    
    # Get the base name of the script's directory
    script_base_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

    # Define the types of data to process and their respective label subdirectories
    data_types_to_process = [
        {"name": "real", "label_subdir": "labels", "suffix": "_real"},
        {"name": "simulation", "label_subdir": "labels", "suffix": "_sim"}
    ]

    for data_type_info in data_types_to_process:
        type_name = data_type_info["name"]
        label_subdir_name = data_type_info["label_subdir"]
        type_suffix = data_type_info["suffix"]
        
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
            empty_output_file = os.path.join(current_output_typed_dir, f'{script_base_name}{type_suffix}_{overall_timestamp}_empty.csv')
            with open(empty_output_file, 'w', newline='') as f:
                fieldnames = ["file_name", "lane", "obstacle", "decision"]
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
        

        for data_item in data_text_current_type:
            # Lane detection
            lane_options = ["left lane", "right lane", "unclear"]
            best_score_for_lane_extraction = -1
            best_matched_phrase_for_lane = ""
            for option in lane_options:
                current_dynamic_query_lane = f"Lane: {option}"
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
                current_dynamic_query_obstacle = f"Obstacles: {option}"
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
                current_dynamic_query_decision = f"Decision: {option}"
                temp_matched_phrase, temp_score = extract_phrase(data_item.lower(), current_dynamic_query_decision.lower())
                if temp_score > best_score_for_decision_extraction:
                    best_score_for_decision_extraction = temp_score
                    best_matched_phrase_for_decision = temp_matched_phrase
            detected_decision, _ = detect_category(best_matched_phrase_for_decision.lower(), decision_options, score_cutoff=50)
            detected_decisions.append(detected_decision)

        results_current_type = []
        for i in range(len(file_names_current_type)):
            result = {
                "file_name": file_names_current_type[i],
                "lane": detected_lanes[i] if i < len(detected_lanes) else "N/a",
                "obstacle": detected_obstacles[i] if i < len(detected_obstacles) else "N/a",
                "decision": detected_decisions[i] if i < len(detected_decisions) else "N/a",
            }
            results_current_type.append(result)
        
        if results_current_type:
            output_csv_filename = f'{script_base_name}{type_suffix}.csv'
            output_csv_filepath = os.path.join(current_output_typed_dir, output_csv_filename)
            with open(output_csv_filepath, 'w', newline='') as f:
                fieldnames = ["file_name", "lane", "obstacle", "decision"]
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

def calculate_precision_recall(conf_matrix, labels):
    """
    Calculates precision, recall, and specificity for each class from a confusion matrix.

    Note:
    - Recall is also known as "Sensitivity" or the True Positive Rate.
    - Specificity is the True Negative Rate.
    - Precision is the measure of a model's accuracy in classifying a sample as positive.

    Args:
        conf_matrix (np.ndarray): The confusion matrix.
        labels (list): The list of class labels, corresponding to the matrix indices.

    Returns:
        pd.DataFrame: A DataFrame with precision, recall, and specificity for each class.
    """
    metrics = []
    total_samples = conf_matrix.sum()
    for i, label in enumerate(labels):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = total_samples - (TP + FP + FN)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0 # also sensitivity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        metrics.append({
            'sub_category': label,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN
        })
    
    return pd.DataFrame(metrics)

def analyze_and_generate_report(model_results_csv_paths_list, overall_timestamp,
                                beta_decision, beta_lane, beta_obstacle,
                                weight_decision, weight_lane, weight_obstacle,
                                obstacle_order):
    """
    Analyzes model predictions against ground truth, generates confusion matrices,
    calculates various metrics, and saves a comprehensive report.
    """
    print(f"\nStarting analysis and report generation using timestamp: {overall_timestamp}...")

    base_comparison_dir = os.path.join(script_dir, "categorical_comparison")
    os.makedirs(base_comparison_dir, exist_ok=True)
    print(f"Analysis reports will be saved in subdirectories of: {base_comparison_dir}")
    
    script_base_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    generated_report_files = []

    for model_results_csv in model_results_csv_paths_list:
        try:
            model_file_parent_dir = os.path.dirname(model_results_csv)
            data_type = os.path.basename(model_file_parent_dir)

            if data_type not in ["real", "simulation"]:
                print(f"  Warning: Could not determine data type for {model_results_csv} from dir '{data_type}'. Skipping.")
                continue

            ground_truth_csv = os.path.join(model_file_parent_dir, f"ground_truth_{data_type}.csv")

            if not os.path.exists(ground_truth_csv):
                print(f"  Warning: Ground truth file not found at {ground_truth_csv}. Skipping comparison for this file.")
                continue

            print(f"\n  Processing analysis for data type: '{data_type}'")
            print(f"    Model results CSV: {model_results_csv}")
            print(f"    Ground truth CSV: {ground_truth_csv}")

            gt_df = pd.read_csv(ground_truth_csv)
            pred_df = pd.read_csv(model_results_csv)
            merged_df = pd.merge(gt_df, pred_df, on='file_name', suffixes=('_gt', '_pred'))

            categories = [col for col in gt_df.columns if col != 'file_name']
            
            report_buffer = io.StringIO()
            category_scores = {}

            for category in categories:
                gt_col = f'{category}_gt'
                pred_col = f'{category}_pred'

                if gt_col not in merged_df.columns or pred_col not in merged_df.columns:
                    report_buffer.write(f"Category '{category}' not found. Skipping.\n\n")
                    continue

                y_true = merged_df[gt_col]
                y_pred = merged_df[pred_col]
                
                if category == 'obstacle':
                    labels = obstacle_order
                else:
                    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

                cm = confusion_matrix(y_true, y_pred, labels=labels) # columns are prediction, rows are ground truth
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)

                report_buffer.write(f"--- Analysis for Category: {category} ---\n\n")
                report_buffer.write("Confusion Matrix:\n")
                report_buffer.write(cm_df.to_string())
                report_buffer.write("\n\n")

                metrics_df = calculate_precision_recall(cm, labels)
                report_buffer.write("Metrics per Sub-category:\n")
                report_buffer.write(metrics_df.to_string(index=False))
                report_buffer.write("\n\n")

                if category == 'decision':
                    f_beta = fbeta_score(y_true, y_pred, beta=beta_decision, average='macro', labels=labels, zero_division=0)
                    category_scores['decision_fbeta'] = f_beta
                    report_buffer.write(f"Macro F-{beta_decision} Score: {f_beta:.4f}\n")
                elif category == 'lane':
                    f_beta = fbeta_score(y_true, y_pred, beta=beta_lane, average='macro', labels=labels, zero_division=0)
                    category_scores['lane_fbeta'] = f_beta
                    report_buffer.write(f"Macro F-{beta_lane} Score: {f_beta:.4f}\n")
                elif category == 'obstacle':
                    f_beta = fbeta_score(y_true, y_pred, beta=beta_obstacle, average='macro', labels=labels, zero_division=0)
                    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=labels)
                    category_scores['obstacle_fbeta'] = f_beta
                    category_scores['obstacle_kappa'] = kappa
                    report_buffer.write(f"Macro F-{beta_obstacle} Score: {f_beta:.4f}\n")
                    report_buffer.write(f"Quadratic Weighted Cohen's Kappa: {kappa:.4f}\n")

                report_buffer.write("\n" + "="*50 + "\n\n")

            # --- Overall Metrics ---
            report_buffer.write("--- Overall Performance Metrics ---\n\n")

            # Complete Exact Accuracy
            correct_predictions = (merged_df[[f'{cat}_gt' for cat in categories]].values == merged_df[[f'{cat}_pred' for cat in categories]].values).all(axis=1).sum()
            complete_exact_accuracy = correct_predictions / len(merged_df)
            report_buffer.write(f"Complete Exact Accuracy: {complete_exact_accuracy:.4f} ({correct_predictions}/{len(merged_df)})\n\n")

            # Composite Score
            decision_fbeta = category_scores.get('decision_fbeta', 0)
            lane_fbeta = category_scores.get('lane_fbeta', 0)
            obstacle_kappa = category_scores.get('obstacle_kappa', 0)
            obstacle_fbeta = category_scores.get('obstacle_fbeta', 0)

            composite_score = (
                decision_fbeta * weight_decision +
                lane_fbeta * weight_lane +
                obstacle_kappa * weight_obstacle
            )
            
            report_buffer.write("--- Components of Composite Score ---\n")
            report_buffer.write(f"Decision Macro F-{beta_decision} Score: {decision_fbeta:.4f}\n")
            report_buffer.write(f"Lane Macro F-{beta_lane} Score: {lane_fbeta:.4f}\n")
            report_buffer.write(f"Obstacle Macro F-{beta_obstacle} Score: {obstacle_fbeta:.4f}\n")
            report_buffer.write(f"Obstacle Quadratic Weighted Cohen's Kappa: {obstacle_kappa:.4f}\n\n")

            report_buffer.write(f"Composite Score: {composite_score:.4f}\n")
            report_buffer.write(f"(Weights: Decision F-beta={weight_decision}, Lane F-beta={weight_lane}, Obstacle Kappa={weight_obstacle})\n")
            
            suffix = '_real' if data_type == 'real' else '_sim'
            output_filename = f"{script_base_name}{suffix}_report.txt"
            output_filepath = os.path.join(base_comparison_dir, output_filename)
            
            with open(output_filepath, 'w') as f:
                f.write(report_buffer.getvalue())

            print(f"  Analysis complete. Report saved to: {output_filepath}")
            generated_report_files.append(output_filepath)

        except FileNotFoundError as e:
            print(f"Error analyzing {model_results_csv}: {e}. Please check file paths.")
        except Exception as e:
            print(f"An unexpected error occurred while analyzing {model_results_csv}: {e}")

    if not generated_report_files:
        print("No analysis reports were generated.")
    else:
        print(f"\nAnalysis process complete. Generated reports: {generated_report_files}")
    
    return generated_report_files

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
            
            # --- User Configuration ---
            base_model_name = os.getenv('MODEL_1.3B_CHKPOINT')

            adapter_checkpoint = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_shortprompt_no-lidar_CE-loss/first_epoch_chkpoint/adapter/"
            merged_model_dir = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/full_b6-s42_shortprompt_no-lidar_CE-loss/first_epoch_chkpoint/merged_model/"
            test_labels_dir = "./eval/"
            extraction_dir = "./extraction_results/"
            
            merge_model_flag = True
            quantize_model_flag = True
            run_inference_flag = True
            extract_cat_flag = True

            # F-beta score parameters (beta > 1 gives more weight to recall, beta < 1 to precision)
            BETA_DECISION = 2.0  # Emphasize recall for Decision
            BETA_LANE = 1.0      # Balanced F1-score for Lane
            BETA_OBSTACLE = 1.0  # Balanced F1-score for Obstacle

            # Composite score weights
            WEIGHT_DECISION = 0.5
            WEIGHT_LANE = 0.3
            WEIGHT_OBSTACLE = 0.2

            # Ordinal labels for Obstacle category for Cohen's Kappa.
            # The order is important: from most to least critical.
            OBSTACLE_ORDER = ["very close", "near", "far away", "not on the same lane"]
            # --------------------------

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
            if run_inference_flag:
                print("\n===== STEP 2: RUNNING INFERENCE ON TEST IMAGES =====")
                inference_results_dir = run_inference(merged_model_path, test_labels_dir, mode="4bit")

            else:
                inference_results_dir = "./eval_output/inference_20250525_000824"

            # Step 3: Extract categories from results
            if extract_cat_flag:
                
                print("\n===== STEP 3: EXTRACTING CATEGORIES FROM RESULTS =====")
                overall_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_results_csv = extract_categories_from_results(inference_results_dir, extraction_dir, overall_timestamp)
            else:
                model_results_csv = ["./extraction_results/real/full_b6-s42_shortfixedprompt_short-lidar_CE-loss_best_real.csv", "./extraction_results/simulation/full_b6-s42_shortfixedprompt_short-lidar_CE-loss_best_sim.csv"]
                overall_timestamp = "20250705_014913"

            # Step 4: Compare results with ground truth
            print("\n===== STEP 4: COMPARING RESULTS WITH GROUND TRUTH =====")
            analyze_and_generate_report(
                model_results_csv, 
                overall_timestamp,
                BETA_DECISION, BETA_LANE, BETA_OBSTACLE,
                WEIGHT_DECISION, WEIGHT_LANE, WEIGHT_OBSTACLE,
                OBSTACLE_ORDER
            )

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
        
if __name__ == "__main__":
    main() 