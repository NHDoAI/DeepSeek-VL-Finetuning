import os
import re
import pandas as pd
import argparse

def parse_report(file_path):
    """
    Parses a single report file to extract evaluation metrics.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    metrics = {}

    # Extract metrics using regex
    accuracy_match = re.search(r"Complete Exact Accuracy: (\d+\.\d+)", content)
    metrics['Complete Exact Accuracy'] = float(accuracy_match.group(1)) if accuracy_match else None

    decision_f2_match = re.search(r"Decision Macro F-2.0 Score: (\d+\.\d+)", content)
    metrics['Decision Macro F-2.0 Score'] = float(decision_f2_match.group(1)) if decision_f2_match else None

    lane_f1_match = re.search(r"Lane Macro F-1.0 Score: (\d+\.\d+)", content)
    metrics['Lane Macro F-1.0 Score'] = float(lane_f1_match.group(1)) if lane_f1_match else None

    obstacle_f1_match = re.search(r"Obstacle Macro F-1.0 Score: (\d+\.\d+)", content)
    metrics['Obstacle Macro F-1.0 Score'] = float(obstacle_f1_match.group(1)) if obstacle_f1_match else None

    obstacle_kappa_match = re.search(r"Obstacle Quadratic Weighted Cohen's Kappa: (\d+\.\d+)", content)
    metrics['Obstacle Quadratic Weighted Cohen\'s Kappa'] = float(obstacle_kappa_match.group(1)) if obstacle_kappa_match else None

    composite_score_match = re.search(r"Composite Score: (\d+\.\d+)", content)
    metrics['Composite Score'] = float(composite_score_match.group(1)) if composite_score_match else None

    weights_match = re.search(r"\(Weights: Decision F-beta=([\d.]+), Lane F-beta=([\d.]+), Obstacle Kappa=([\d.]+)\)", content)
    if weights_match:
        metrics['Weight Decision F-beta'] = float(weights_match.group(1))
        metrics['Weight Lane F-beta'] = float(weights_match.group(2))
        metrics['Weight Obstacle Kappa'] = float(weights_match.group(3))
    else:
        metrics['Weight Decision F-beta'] = None
        metrics['Weight Lane F-beta'] = None
        metrics['Weight Obstacle Kappa'] = None

    # Get model name from filename
    basename = os.path.basename(file_path)
    model_name = basename.replace('_real_report.txt', '').replace('_sim_report.txt', '')
    metrics['model'] = model_name

    return metrics

def process_folder(folder_path, output_csv):
    """
    Processes all report files in a folder and its subfolders.
    """
    all_metrics = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_real_report.txt") or file.endswith("_sim_report.txt"):
                file_path = os.path.join(root, file)
                metrics = parse_report(file_path)
                all_metrics.append(metrics)

    if not all_metrics:
        print("No report files found in the specified folder.")
        return

    df = pd.DataFrame(all_metrics)
    
    # Reorder columns and set index
    cols = ['model', 'Complete Exact Accuracy', 'Decision Macro F-2.0 Score', 'Lane Macro F-1.0 Score',
            'Obstacle Macro F-1.0 Score', "Obstacle Quadratic Weighted Cohen's Kappa", 'Composite Score',
            'Weight Decision F-beta', 'Weight Lane F-beta', 'Weight Obstacle Kappa']
    
    # Ensure all columns exist, fill missing with None
    for col in cols:
        if col not in df.columns:
            df[col] = None
            
    df = df[cols]
    df = df.set_index('model')

    df.to_csv(output_csv)
    print(f"Successfully created CSV file: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics from report files and save to CSV.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing report files.")
    parser.add_argument("--output_csv", type=str, default="evaluation_metrics.csv", help="The name of the output CSV file.")
    args = parser.parse_args()

    process_folder(args.folder_path, args.output_csv) 