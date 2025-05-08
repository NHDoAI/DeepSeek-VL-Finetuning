"""
compare the responses of two models on the same set of images as well as the ground truth, then save the results to a file. Used to investigate why one model has better loss but worse accuracy.
"""

import json
import os
import glob
from pathlib import Path
import difflib

def extract_assistant_content(json_file_path):
    """Extract the content from the assistant's response in a JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data["conversation"][1]["content"]

def compare_files():
    # Directory paths
    ground_truth_dir = "./1.3B_less-lora_first-epoch/test/labels"
    model1_dir = "./1.3B_less-lora_first-epoch/eval_output/inference_20250420_174540"
    model2_dir = "./1.3B_less-lora_later-save/eval_output/inference_20250420_145442"
    
    # Get all ground truth filenames
    ground_truth_files = glob.glob(f"{ground_truth_dir}/*.json")
    
    results = []
    
    # Create output file
    with open("comparison_results.txt", "w") as outfile:
        outfile.write("# Comparison Results for Model Outputs\n\n")
        
        for gt_file in ground_truth_files:
            filename = os.path.basename(gt_file)
            
            # Check if corresponding files exist in model output directories
            model1_file = os.path.join(model1_dir, filename)
            model2_file = os.path.join(model2_dir, filename)
            
            if not os.path.exists(model1_file) or not os.path.exists(model2_file):
                outfile.write(f"Warning: Missing corresponding file for {filename}\n\n")
                continue
            
            # Extract content from each file
            gt_content = extract_assistant_content(gt_file)
            model1_content = extract_assistant_content(model1_file)
            model2_content = extract_assistant_content(model2_file)
            
            # Calculate similarity ratio for comparison
            model1_similarity = difflib.SequenceMatcher(None, gt_content, model1_content).ratio()
            model2_similarity = difflib.SequenceMatcher(None, gt_content, model2_content).ratio()
            models_similarity = difflib.SequenceMatcher(None, model1_content, model2_content).ratio()
            
            # Write comparison to file
            outfile.write(f"## File: {filename}\n\n")
            
            outfile.write("### Ground Truth Content:\n")
            outfile.write(f"```\n{gt_content}\n```\n\n")
            
            outfile.write("### Model 1 Content:\n")
            outfile.write(f"```\n{model1_content}\n```\n\n")
            
            outfile.write("### Model 2 Content:\n")
            outfile.write(f"```\n{model2_content}\n```\n\n")
            
            outfile.write("### Similarity Statistics:\n")
            outfile.write(f"- Model 1 similarity to ground truth: {model1_similarity:.4f}\n")
            outfile.write(f"- Model 2 similarity to ground truth: {model2_similarity:.4f}\n")
            outfile.write(f"- Model 1 similarity to Model 2: {models_similarity:.4f}\n\n")
            outfile.write("-" * 80 + "\n\n")
            
            results.append({
                "filename": filename,
                "model1_similarity": model1_similarity,
                "model2_similarity": model2_similarity,
                "models_similarity": models_similarity
            })
    
    return results

def main():
    results = compare_files()
    
    # Calculate average similarity
    avg_model1_sim = sum(r["model1_similarity"] for r in results) / len(results)
    avg_model2_sim = sum(r["model2_similarity"] for r in results) / len(results)
    avg_models_sim = sum(r["models_similarity"] for r in results) / len(results)
    
    # Print summary to console
    print(f"Analyzed {len(results)} file sets")
    print(f"Model 1 average similarity to ground truth: {avg_model1_sim:.4f}")
    print(f"Model 2 average similarity to ground truth: {avg_model2_sim:.4f}")
    print(f"Model 1 average similarity to Model 2: {avg_models_sim:.4f}")
    print(f"Detailed comparison results saved to comparison_results.txt")
    
    # Append summary to the results file
    with open("comparison_results.txt", "a") as outfile:
        outfile.write("# Summary Statistics\n\n")
        outfile.write(f"Analyzed {len(results)} file sets\n")
        outfile.write(f"Model 1 average similarity to ground truth: {avg_model1_sim:.4f}\n")
        outfile.write(f"Model 2 average similarity to ground truth: {avg_model2_sim:.4f}\n")
        outfile.write(f"Model 1 average similarity to Model 2: {avg_models_sim:.4f}\n")

if __name__ == "__main__":
    main() 