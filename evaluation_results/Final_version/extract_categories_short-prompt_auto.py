"""
How to use: python3 extract_categories_auto.py /path/to/your/First-run_one-epoch_b6_long-lidar
"""
import os
import json
import csv
import argparse
from rapidfuzz import fuzz
import pandas as pd

# ============ HELPER FUNCTIONS FOR TEXT EXTRACTION (from manual script) ============

def extract_phrase(text_target: str, phrase_query: str, score_cutoff: int = 0) -> (str, float):
    """
    Extract a phrase from target text using fuzzy matching.
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
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {json_file}. File might be empty or malformed.")
        return None
    except Exception as e:
        print(f"Warning: Error reading file {json_file}: {e}")
        return None

# ============ CATEGORY EXTRACTION LOGIC ============

def extract_categories_from_text(text_content: str) -> dict:
    """
    Extracts all categories from a single text content.
    """
    # Lane detection
    lane_options = ["left lane", "right lane", "unclear"]
    best_score_for_lane_extraction = -1
    best_matched_phrase_for_lane = ""
    for option in lane_options:
        current_dynamic_query_lane = f"Lane: {option}"
        temp_matched_phrase, temp_score = extract_phrase(text_content.lower(), current_dynamic_query_lane.lower())
        if temp_score > best_score_for_lane_extraction:
            best_score_for_lane_extraction = temp_score
            best_matched_phrase_for_lane = temp_matched_phrase
    detected_lane, _ = detect_category(best_matched_phrase_for_lane.lower(), lane_options, score_cutoff=50)

    # Obstacle detection
    obstacles_options = ["far away", "near", "very close", "not on the same lane"]
    best_score_for_obstacle_extraction = -1
    best_matched_phrase_for_obstacle = ""
    for option in obstacles_options:
        current_dynamic_query_obstacle = f"Obstacles: {option}"
        temp_matched_phrase, temp_score = extract_phrase(text_content.lower(), current_dynamic_query_obstacle.lower())
        if temp_score > best_score_for_obstacle_extraction:
            best_score_for_obstacle_extraction = temp_score
            best_matched_phrase_for_obstacle = temp_matched_phrase
    detected_obstacle, _ = detect_category(best_matched_phrase_for_obstacle.lower(), obstacles_options, score_cutoff=50)

    # Decision detection
    decision_options = ["straight forward", "slow cruise", "switch lane"]
    best_score_for_decision_extraction = -1
    best_matched_phrase_for_decision = ""
    for option in decision_options:
        current_dynamic_query_decision = f"Decision: {option}"
        temp_matched_phrase, temp_score = extract_phrase(text_content.lower(), current_dynamic_query_decision.lower())
        if temp_score > best_score_for_decision_extraction:
            best_score_for_decision_extraction = temp_score
            best_matched_phrase_for_decision = temp_matched_phrase
    detected_decision, _ = detect_category(best_matched_phrase_for_decision.lower(), decision_options, score_cutoff=50)


    return {
        "lane": detected_lane,
        "obstacle": detected_obstacle,
        "decision": detected_decision,

    }

# ============ CORE PROCESSING FUNCTION ============

def process_directory(root_dir: str):
    """
    Traverses the directory structure, finds JSON files, extracts data, and saves to CSV.
    """
    print(f"Starting processing in root directory: {root_dir}")

    for model_run_dir_name in os.listdir(root_dir):
        model_run_dir_path = os.path.join(root_dir, model_run_dir_name)
        if not os.path.isdir(model_run_dir_path):
            continue

        eval_output_path = os.path.join(model_run_dir_path, "eval_output")
        if not os.path.isdir(eval_output_path):
            continue
        
        print(f"\nFound eval_output in: {model_run_dir_name}")
        
        for root, dirs, _ in os.walk(eval_output_path):
            if "real" in dirs:
                process_type_folder(model_run_dir_name, model_run_dir_path, os.path.join(root, "real"), "real")
                dirs.remove("real") # Avoid re-visiting
            
            if "simulation" in dirs:
                process_type_folder(model_run_dir_name, model_run_dir_path, os.path.join(root, "simulation"), "sim")
                dirs.remove("simulation") # Avoid re-visiting


def process_type_folder(model_run_dir_name, model_run_dir_path, type_path, type_name):
    """
    Processes a 'real' or 'simulation' folder to find JSONs and extract data.
    """
    print(f"  Processing '{type_name}' data...")
    json_files = []
    for root, _, files in os.walk(type_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"    No JSON files found for '{type_name}' in {model_run_dir_name}.")
        return

    print(f"    Found {len(json_files)} JSON files.")
    
    all_results = []
    for json_file in json_files:
        json_data = extract_text_from_json(json_file)
        if json_data and isinstance(json_data.get("conversation"), list) and len(json_data["conversation"]) > 1:
            text_content = json_data["conversation"][1].get("content", "")
            if text_content:
                extracted_data = extract_categories_from_text(text_content)
                extracted_data["file_name"] = os.path.basename(json_file)
                all_results.append(extracted_data)
            else:
                print(f"Warning: No content found in conversation for {json_file}")
        else:
            print(f"Warning: Could not find model response in {json_file}. Skipping.")

    if not all_results:
        print(f"    No data extracted for '{type_name}' in {model_run_dir_name}.")
        return

    # Create output directory
    output_dir_base = os.path.join(model_run_dir_path, "extraction_results")
    output_dir_type = os.path.join(output_dir_base, type_name if type_name == "real" else "simulation")

    os.makedirs(output_dir_type, exist_ok=True)

    # Define CSV file path and name
    csv_file_name = f"{model_run_dir_name}_{type_name}.csv"
    csv_file_path = os.path.join(output_dir_type, csv_file_name)

    # Write to CSV
    df = pd.DataFrame(all_results)
    fieldnames = ["file_name", "lane", "obstacle", "decision"]
    df = df[fieldnames] # Ensure column order
    df.to_csv(csv_file_path, index=False)
    
    print(f"    Successfully saved extracted data to {csv_file_path}")


# ============ MAIN EXECUTION BLOCK ============

def main():
    """
    Main function to parse command-line arguments and run the extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Automatically find and process JSON evaluation results from a directory structure."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="The root directory containing model run folders (e.g., 'First-run_one-epoch_b6_long-lidar').",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: The specified root directory does not exist: {args.root_dir}")
        return

    process_directory(args.root_dir)
    print("\nProcessing complete.")

if __name__ == "__main__":
    main() 