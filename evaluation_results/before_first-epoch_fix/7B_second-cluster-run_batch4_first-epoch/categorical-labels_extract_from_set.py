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
    #test_labels_dir = os.getenv("TEST_LABELS_PATH")
    test_labels_dir = "./eval_output/inference_20250418_125452" #model
    output_dir = 'extraction_results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = [os.path.join(test_labels_dir, fname) for fname in os.listdir(test_labels_dir) if fname.endswith('.json')]
    
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
        detected_lane, score = detect_category(matched_phrase.lower(), lane_options, score_cutoff = 76)
        detected_lanes.append(detected_lane)

        # Obstacle detection
        current_query = "Obstacles are not on the same lane"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        obstacles_options = ["far away", "near", "very close", "not on the same lane"]
        detected_obstacle, score = detect_category(matched_phrase.lower(), obstacles_options, score_cutoff = 76)
        detected_obstacles.append(detected_obstacle)

        # Decision detection
        current_query = "optimal movement decision is: change lane left"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        decision_options = ["straight forward", "slow cruise", "change lane left", "change lane right"]
        detected_decision, score = detect_category(matched_phrase.lower(), decision_options, score_cutoff = 76)
        detected_decisions.append(detected_decision)

        # Final decision detection
        current_query = "[Final Decision]\nchange lane left"
        matched_phrase, score = extract_phrase(data.lower(), current_query.lower())
        
        final_key_index = matched_phrase.rfind("[final")# Find the position of the word "Final" in the string

        # Slice the string from that position
        if final_key_index != -1:  # Ensure "Final" exists in the string
            matched_phrase = matched_phrase[final_key_index:]
        else:
            print("The keyword 'Final' was not found in the string.")

        final_decision_options = ["straight forward", "slow cruise", "change lane left", "change lane right"]
        detected_final_decision, score = detect_category(matched_phrase.lower(), final_decision_options, score_cutoff = 76)
        detected_final_decisions.append(detected_final_decision)
        

    # Create a dictionary to store all extracted information
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
    
    # Save results to CSV file
    output_file = os.path.join(output_dir, 'extraction_results_model-first-epoch.csv')
    with open(output_file, 'w', newline='') as f:
        fieldnames = ["file_name", "lane", "obstacle", "decision", "final_decision"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Extraction complete. Results saved to {output_file}")

if __name__ == "__main__":
    main() 