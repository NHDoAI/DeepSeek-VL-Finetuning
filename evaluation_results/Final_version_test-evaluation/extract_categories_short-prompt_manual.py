"""
How to use: python3 extract_categories_short-prompt_manual.py -i ./eval_output/inference_20250613_034437/ -o ./manual_extract
"""

import os
import json
import csv
import argparse
from datetime import datetime
from rapidfuzz import fuzz

# ============ HELPER FUNCTIONS FOR TEXT EXTRACTION ============

def extract_phrase(text_target: str, phrase_query: str, score_cutoff: int = 0) -> (str, float):
    """
    Extract a phrase from target text using fuzzy matching.

    Args:
        text_target (str): The text to search within.
        phrase_query (str): The phrase to search for.
        score_cutoff (int): The minimum score to consider a match.

    Returns:
        tuple: A tuple containing the best matching phrase and its score.
    """
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

def detect_category(text_target: str, categories_query: list, score_cutoff: int = 75) -> (str, float):
    """
    Detect category from text using fuzzy matching against a list of categories.

    Args:
        text_target (str): The text to search within.
        categories_query (list): A list of possible category strings.
        score_cutoff (int): The minimum score to consider a match.

    Returns:
        tuple: A tuple containing the detected category and its score.
    """
    best_score = 0
    detected_category = "N/a"
    for category in categories_query:
        score = fuzz.partial_ratio(text_target, category, score_cutoff=score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            detected_category = category
    return detected_category, best_score

def extract_text_from_json(json_file: str) -> dict:
    """
    Extract text content from a JSON file.

    Args:
        json_file (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# ============ CATEGORY EXTRACTION FUNCTION ============

def extract_categories_from_results(results_dir: str, output_dir: str):
    """
    Extract category information from JSON results and save them to CSV files.

    This function processes subdirectories (e.g., 'real', 'simulation') within
    the main results directory, extracts structured data from model-generated text
    in JSON files, and saves the extracted categories into separate CSV files
    for each data type.

    Args:
        results_dir (str): The path to the directory containing the model's
                           inference results, with 'real' and 'simulation'
                           subdirectories.
        output_dir (str): The path to the directory where the output CSV files
                          will be saved.
    """
    overall_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting category extraction process with timestamp: {overall_timestamp}...")
    os.makedirs(output_dir, exist_ok=True)

    generated_csv_files = []

    data_types_to_process = [
        {"name": "real", "label_subdir": "labels"},
        {"name": "simulation", "label_subdir": "labels"}
    ]

    for data_type_info in data_types_to_process:
        type_name = data_type_info["name"]
        label_subdir_name = data_type_info["label_subdir"]
        
        print(f"\nProcessing data type: '{type_name}'")

        current_input_labels_dir = os.path.join(results_dir, type_name, label_subdir_name)
        
        if not os.path.isdir(current_input_labels_dir):
            print(f"Warning: Input directory not found for '{type_name}': {current_input_labels_dir}. Skipping.")
            continue

        json_label_files = [
            os.path.join(current_input_labels_dir, fname)
            for fname in os.listdir(current_input_labels_dir)
            if fname.endswith('.json')
        ]

        if not json_label_files:
            print(f"No JSON label files found in {current_input_labels_dir}. Skipping.")
            continue
        
        print(f"Found {len(json_label_files)} JSON files to process for '{type_name}'.")
        
        data_text_current_type = []
        file_names_current_type = []
        
        for file_path in json_label_files:
            try:
                data = extract_text_from_json(file_path)
                if len(data.get("conversation", [])) > 1 and "content" in data["conversation"][1]:
                    data_text_current_type.append(data["conversation"][1]["content"])
                    file_names_current_type.append(os.path.basename(file_path))
                else:
                    print(f"Warning: Could not find model response in {file_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}. Skipping.")

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
                "decision": detected_decisions[i] if i < len(detected_decisions) else "N/a"
            }
            results_current_type.append(result)
        
        if results_current_type:
            # Create a dedicated subdirectory for the data type in the output folder
            type_output_dir = os.path.join(output_dir, type_name)
            os.makedirs(type_output_dir, exist_ok=True)
            
            output_csv_filename = f'extraction_results_model_{type_name}_{overall_timestamp}.csv'
            output_csv_filepath = os.path.join(type_output_dir, output_csv_filename)
            
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
        print("\nNo CSV files were generated in this run.")
    
    print(f"\nExtraction process complete. Generated CSVs: {generated_csv_files}")

# ============ MAIN EXECUTION BLOCK ============

def main():
    """
    Main function to parse command-line arguments and run the extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Extract categories from model prediction JSON files and save to CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input_dir',
        required=True,
        help="Path to the directory containing the model's prediction results.\n"
             "This directory should contain subdirectories like 'real/labels/' and/or 'simulation/labels/'."
    )
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help="Path to the directory where the output CSV files will be saved."
    )

    args = parser.parse_args()

    # Run the extraction process
    extract_categories_from_results(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 