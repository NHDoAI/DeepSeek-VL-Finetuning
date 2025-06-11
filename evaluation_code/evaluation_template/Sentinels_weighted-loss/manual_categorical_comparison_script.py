"""
Manually extract categories from model inference results and ground truth CSV files, then compare them, show rows where predictions are wrong.

How to use:
1. Set the model_results_file and ground_truth_file paths in the main() function.
2. Set the data_type_for_run and output_directory variables in the main() function.
3. Run the script.
"""
import os
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Define standard category orders (can be customized)
CATEGORY_ORDERS = {
    'lane': ["left lane", "right lane", "unclear"],
    'obstacle': ["far away", "near", "very close", "not on the same lane"],
    'decision': ["straight forward", "slow cruise", "switch lane"],
    'final_decision': ["straight forward", "slow cruise", "switch lane"]
}

def perform_comparison_and_generate_reports(
    model_results_csv_path: str,
    ground_truth_csv_path: str,
    data_type: str, # e.g., "real", "simulation"
    output_base_dir: str,
    timestamp: str
):
    """
    Compares model-extracted categories with ground truth from CSV files.
    Outputs comparison_stats and category_counts CSV files with specified formatting.
    """
    print(f"\nStarting comparison for data type: '{data_type}' with timestamp: {timestamp}...")
    print(f"  Model results CSV: {model_results_csv_path}")
    print(f"  Ground truth CSV: {ground_truth_csv_path}")

    # Define the specific output directory for this data_type's comparison results
    type_specific_comparison_output_dir = os.path.join(output_base_dir, data_type)
    os.makedirs(type_specific_comparison_output_dir, exist_ok=True)
    print(f"  Comparison reports will be saved in: {type_specific_comparison_output_dir}")

    if not os.path.exists(ground_truth_csv_path):
        print(f"  Error: Ground truth file not found at {ground_truth_csv_path}. Skipping comparison.")
        return None, None
    
    if not os.path.exists(model_results_csv_path):
        print(f"  Error: Model results file not found at {model_results_csv_path}. Skipping comparison.")
        return None, None

    try:
        df_ground_truth = pd.read_csv(ground_truth_csv_path)
        df_model = pd.read_csv(model_results_csv_path)
    except Exception as e:
        print(f"  Error reading CSV files for {data_type}: {e}. Skipping this comparison.")
        return None, None
    
    if df_ground_truth.empty:
        print(f"  Error: Ground truth CSV ({ground_truth_csv_path}) is empty. Skipping.")
        return None, None
    if df_model.empty:
        print(f"  Error: Model results CSV ({model_results_csv_path}) is empty. Skipping.")
        return None, None

    # Get ID column (first column) and other columns to compare
    if not df_ground_truth.columns.empty:
        id_column = df_ground_truth.columns[0]
    else:
        print(f"  Error: Ground truth CSV ({ground_truth_csv_path}) has no columns. Skipping.")
        return None, None
        
    columns_to_compare = [col for col in df_ground_truth.columns if col != id_column]
    
    if not columns_to_compare:
        print(f"  Warning: No columns to compare (excluding ID column '{id_column}') in ground truth CSV. Only ID column found.")
        # Proceeding might result in empty reports, or you can choose to return early.
        # For now, let's allow it to proceed and generate empty reports if applicable.

    comparison_results_stats = {} 
    category_counts_data = {}
    sub_category_comparison_stats_for_current_file = []
    
    for column in columns_to_compare:
        if column not in df_model.columns:
            print(f"  Warning: Column '{column}' not found in model results CSV. Skipping comparison for this column.")
            continue
        if column not in df_ground_truth.columns: # Should not happen if columns_to_compare is from df_ground_truth
            print(f"  Warning: Column '{column}' not found in ground truth CSV (this should not happen). Skipping comparison for this column.")
            continue

        gt_col_name = f"{column}_ground_truth"
        model_col_name = f"{column}_model"
        
        merged_df = pd.merge(
            df_ground_truth[[id_column, column]], 
            df_model[[id_column, column]],
            on=id_column,
            how='inner', # Use 'inner' to only compare common IDs; use 'outer' for all IDs from both
            suffixes=('_ground_truth', '_model')
        )
        
        if merged_df.empty:
            print(f"  Warning: No common IDs ('{id_column}') found for column '{column}' between model and ground truth for {data_type}. Skipping this column.")
            continue

        merged_df['is_same'] = merged_df[gt_col_name] == merged_df[model_col_name]
        
        total_rows = len(merged_df)
        matching_rows = merged_df['is_same'].sum()
        non_matching_rows = total_rows - matching_rows
        match_percentage = (matching_rows / total_rows * 100) if total_rows > 0 else 0
        
        comparison_results_stats[column] = {
            'total_rows': total_rows,
            'matching_rows': matching_rows,
            'non_matching_rows': non_matching_rows,
            'match_percentage': match_percentage
        }

        if column in CATEGORY_ORDERS:
            for sub_category_value in CATEGORY_ORDERS[column]:
                sub_category_df = merged_df[merged_df[gt_col_name] == sub_category_value]
                sub_total_rows = len(sub_category_df)
                
                if sub_total_rows > 0:
                    sub_matching_rows = sub_category_df['is_same'].sum()
                    sub_non_matching_rows = sub_total_rows - sub_matching_rows
                    sub_match_percentage = (sub_matching_rows / sub_total_rows * 100)
                else:
                    sub_matching_rows = 0
                    sub_non_matching_rows = 0
                    sub_match_percentage = 0.0

                sub_category_name_for_csv = f"{column}:{sub_category_value}"
                sub_category_comparison_stats_for_current_file.append({
                    'category_name': sub_category_name_for_csv,
                    'total_rows': sub_total_rows,
                    'matching_rows': sub_matching_rows,
                    'non_matching_rows': sub_non_matching_rows,
                    'match_percentage': sub_match_percentage
                })
        
        category_counts_data[column] = {
            'ground_truth': merged_df[gt_col_name].value_counts().to_dict(),
            'model': merged_df[model_col_name].value_counts().to_dict()
        }
    
    # Save comparison_stats.csv
    comparison_stats_file = os.path.join(type_specific_comparison_output_dir, f'comparison_stats_{data_type}_{timestamp}.csv')
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
        if sub_category_comparison_stats_for_current_file: # Add an empty line before sub-category stats if they exist
            writer.writerow([])
        for stats in sub_category_comparison_stats_for_current_file:
            writer.writerow([
                stats['category_name'],
                stats['total_rows'],
                stats['matching_rows'],
                stats['non_matching_rows'],
                f"{stats['match_percentage']:.2f}%"
            ])
    print(f"  Comparison statistics saved to {comparison_stats_file}")

    # Save category_counts.csv with new formatting
    category_counts_file = os.path.join(type_specific_comparison_output_dir, f'category_counts_{data_type}_{timestamp}.csv')
    with open(category_counts_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Type', 'Value', 'Count'])

        # --- Write Ground Truth Data ---
        print(f"    Writing ground truth counts to {category_counts_file}...")
        gt_data_written = False
        for i, category_key in enumerate(columns_to_compare):
            if category_key in category_counts_data and 'ground_truth' in category_counts_data[category_key]:
                gt_data_written = True
                ordered_values = CATEGORY_ORDERS.get(category_key, [])
                counts = category_counts_data[category_key]['ground_truth']
                
                # Write values in specified order
                for value in ordered_values:
                    count = counts.get(value, 0)
                    writer.writerow([category_key, 'ground_truth', value, count])
                
                # Write any other values not in the predefined order
                for value, count in counts.items():
                    if value not in ordered_values:
                         writer.writerow([category_key, 'ground_truth', value, count])
                
                # Add empty row if not the last category in the ground_truth block
                if i < len(columns_to_compare) - 1:
                    writer.writerow([]) 
            elif category_key in columns_to_compare: # Category exists but no GT data for it after merge
                 print(f"    Note: No ground truth data for category '{category_key}' after merging. It might be empty or not present in merged data.")


        # Add a separator empty row if both GT and Model data will be present
        model_data_will_be_written = any(
            category_key in category_counts_data and 'model' in category_counts_data[category_key] 
            for category_key in columns_to_compare
        )
        if gt_data_written and model_data_will_be_written:
            writer.writerow([]) 

        # --- Write Model Data ---
        print(f"    Writing model counts to {category_counts_file}...")
        for i, category_key in enumerate(columns_to_compare):
            if category_key in category_counts_data and 'model' in category_counts_data[category_key]:
                ordered_values = CATEGORY_ORDERS.get(category_key, [])
                counts = category_counts_data[category_key]['model']

                # Write values in specified order
                for value in ordered_values:
                    count = counts.get(value, 0)
                    writer.writerow([category_key, 'model', value, count])

                # Write any other values not in the predefined order
                for value, count in counts.items():
                    if value not in ordered_values:
                        writer.writerow([category_key, 'model', value, count])
                
                # Add empty row if not the last category in the model block
                if i < len(columns_to_compare) - 1:
                    writer.writerow([])
            elif category_key in columns_to_compare: # Category exists but no Model data for it after merge
                 print(f"    Note: No model data for category '{category_key}' after merging. It might be empty or not present in merged data.")

    print(f"  Category counts saved to {category_counts_file}")
    
    return comparison_stats_file, category_counts_file

def main():
    """Main function to run the comparison."""
    print("Starting categorical comparison script...")

    # --- Configuration ---
    # Replace with your actual file paths
    # Example: model_results_file = "./extraction_results/real/extraction_results_model_real_20250525_010633.csv"
    # Example: ground_truth_file = "./extraction_results/real/ground_truth_real.csv"
    
    model_results_file = "./extraction_results/simulation/extraction_results_model_simulation_20250601_152454.csv" # IMPORTANT: Update this path
    ground_truth_file = "./extraction_results/simulation/ground_truth_simulation.csv"       # IMPORTANT: Update this path
    
    # data_type helps in organizing outputs, e.g., "real", "simulation", "test_set_A"
    # This could be inferred from file names/paths or set manually.
    data_type_for_run = "simulation" # IMPORTANT: Update as needed
    
    # Base directory where 'data_type_for_run' subfolder will be created for reports
    output_directory = "./manual_comparison_reports"                 # IMPORTANT: Update as needed

    # --- End Configuration ---

    if not os.path.exists(model_results_file) or not os.path.exists(ground_truth_file):
        print(f"\nError: One or both input CSV files do not exist. Please check the paths:")
        if not os.path.exists(model_results_file):
            print(f"  Model results file not found: {os.path.abspath(model_results_file)}")
        if not os.path.exists(ground_truth_file):
            print(f"  Ground truth file not found: {os.path.abspath(ground_truth_file)}")
        print("Please update the 'model_results_file' and 'ground_truth_file' variables in the script.")
        return

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_directory, exist_ok=True)

    stats_file, counts_file = perform_comparison_and_generate_reports(
        model_results_csv_path=model_results_file,
        ground_truth_csv_path=ground_truth_file,
        data_type=data_type_for_run,
        output_base_dir=output_directory,
        timestamp=current_timestamp
    )

    if stats_file and counts_file:
        print(f"\nComparison process complete for data type '{data_type_for_run}'.")
        print(f"  Stats report: {os.path.abspath(stats_file)}")
        print(f"  Counts report: {os.path.abspath(counts_file)}")
    else:
        print(f"\nComparison process for '{data_type_for_run}' encountered errors or produced no output.")
    
    print("\nScript finished.")

if __name__ == "__main__":
    # Before running, ensure your model_results_file and ground_truth_file paths
    # in the main() function are correctly set to your actual CSV files.
    main()