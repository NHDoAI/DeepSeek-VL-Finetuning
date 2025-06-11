"""Extracting and saving the categorical data from json result files. Used for both ground truth and model output"""

import json
import os
import glob
from typing import List, Dict, Any, Set
from rapidfuzz import fuzz
from dotenv import load_dotenv
import json
import csv

def extract_phrase(text_target, phrase_query, score_cutoff = 0):
    split_target = text_target.split()
    split_query = phrase_query.split()
    query_word_count = len(split_query)
    target_word_count = len(split_target)
    best_score = 0
    best_match = ""
    for i in range(target_word_count + 1 - query_word_count):
        candidate = " ".join(split_target[i:i+query_word_count])
        score = fuzz.ratio(candidate, phrase_query, score_cutoff = score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            best_match = candidate
    return best_match, best_score

def detect_category(text_target, categories_query, score_cutoff = 75):
    best_score = 0
    detected_category = "N/a"
    for category in categories_query:
        score = fuzz.partial_ratio(text_target, category, score_cutoff = score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            detected_category = category
    return detected_category, best_score

def extract_text_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    return data

def main():
    # Define keywords to extract - you can modify this list
    load_dotenv()
    # test_labels_dir = os.getenv("TEST_LABELS_PATH")
    # The path below should now be the base directory containing "real" and "simulation" subfolders
    base_test_labels_dir = "./eval_output/inference_20250418_132333" 
    output_dir = 'extraction_results' # This will be the parent directory for "real" and "simulation" outputs
    
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    sub_directories_to_process = ["real", "simulation"]

    for sub_dir_name in sub_directories_to_process:
        current_input_dir = os.path.join(base_test_labels_dir, sub_dir_name)

        if not os.path.isdir(current_input_dir):
            print(f"Warning: Input subdirectory '{current_input_dir}' not found. Skipping.")
            continue

        test_files = [os.path.join(current_input_dir, fname) for fname in os.listdir(current_input_dir) if fname.endswith('.json')]
        
        if not test_files:
            print(f"No JSON files found in '{current_input_dir}'. Skipping this subdirectory.")
            continue
        
        print(f"Processing files from: {current_input_dir}")

        data_text_for_subdir = []
        file_names_for_subdir = []

        for file_path in test_files:
            try:
                data = extract_text_from_json(file_path)
                if "conversation" in data and \
                   isinstance(data["conversation"], list) and \
                   len(data["conversation"]) > 1 and \
                   isinstance(data["conversation"][1], dict) and \
                   "content" in data["conversation"][1]:
                    data_text_for_subdir.append(data["conversation"][1]["content"])
                    file_names_for_subdir.append(os.path.basename(file_path))
                else:
                    print(f"Warning: JSON file {file_path} has unexpected or incomplete structure in 'conversation' data. Skipping this file.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {file_path}. Skipping this file.")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}. Skipping this file.")
        
        if not data_text_for_subdir:
            print(f"No valid data extracted from JSON files in '{current_input_dir}'. No output file will be created for this subdirectory.")
            continue

        detected_lanes_list = []
        detected_obstacles_list = []
        detected_decisions_list = []
        detected_final_decisions_list = []

        for text_content in data_text_for_subdir:
            # Lane detection
            current_query = "The current lane is <LANE> right lane"
            matched_phrase, _ = extract_phrase(text_content.lower(), current_query.lower())
            lane_options = ["left lane", "right lane"]
            detected_lane, _ = detect_category(matched_phrase.lower(), lane_options, score_cutoff = 76)
            detected_lanes_list.append(detected_lane)

            # Obstacle detection
            current_query = "Obstacles are not on the same lane"
            matched_phrase, _ = extract_phrase(text_content.lower(), current_query.lower())
            obstacles_options = ["far away", "near", "very close", "not on the same lane"]
            detected_obstacle, _ = detect_category(matched_phrase.lower(), obstacles_options, score_cutoff = 76)
            detected_obstacles_list.append(detected_obstacle)

            # Decision detection
            current_query = "optimal movement decision is: change lane left"
            matched_phrase, _ = extract_phrase(text_content.lower(), current_query.lower())
            decision_options = ["straight forward", "slow cruise", "change lane left", "change lane right"]
            detected_decision, _ = detect_category(matched_phrase.lower(), decision_options, score_cutoff = 76)
            detected_decisions_list.append(detected_decision)

            # Final decision detection
            current_query = "[Final Decision]\nchange lane left"
            matched_phrase, _ = extract_phrase(text_content.lower(), current_query.lower())
            
            final_key_index = matched_phrase.rfind("[final")
            if final_key_index != -1:
                matched_phrase = matched_phrase[final_key_index:]
            # else:
            #     print(f"Debug: Keyword '[final' not found in matched phrase for final decision from content starting with: {text_content[:50]}...")

            final_decision_options = ["straight forward", "slow cruise", "change lane left", "change lane right"]
            detected_final_decision, _ = detect_category(matched_phrase.lower(), final_decision_options, score_cutoff = 76)
            detected_final_decisions_list.append(detected_final_decision)
            
        # Create result dictionaries for the current subdirectory
        results_for_this_subdir = []
        for i in range(len(file_names_for_subdir)):
            result = {
                "file_name": file_names_for_subdir[i],
                # "source" column is removed as file location implies source
                "lane": detected_lanes_list[i] if i < len(detected_lanes_list) else "N/a",
                "obstacle": detected_obstacles_list[i] if i < len(detected_obstacles_list) else "N/a",
                "decision": detected_decisions_list[i] if i < len(detected_decisions_list) else "N/a",
                "final_decision": detected_final_decisions_list[i] if i < len(detected_final_decisions_list) else "N/a"
            }
            results_for_this_subdir.append(result)
    
        # Save results for the current subdirectory to its own CSV file
        if results_for_this_subdir:
            # Create specific output subdirectory (e.g., extraction_results/real)
            specific_output_subdir = os.path.join(output_dir, sub_dir_name)
            os.makedirs(specific_output_subdir, exist_ok=True)

            output_csv_filename = f'extraction_results_{sub_dir_name}.csv'
            output_csv_path = os.path.join(specific_output_subdir, output_csv_filename)
            
            fieldnames = ["file_name", "lane", "obstacle", "decision", "final_decision"]
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_for_this_subdir)
            
            print(f"Extraction complete for '{sub_dir_name}'. Results saved to {output_csv_path}")
        else:
            # This case is already handled by the check for `data_text_for_subdir`
            # but kept here for logical completeness if that check were different.
            print(f"No data to write for subdirectory '{sub_dir_name}'.")

    print("All processing finished.")

if __name__ == "__main__":
    main() 