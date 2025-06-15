import csv
import os

def get_filenames_from_csv(filepath):
    """
    Reads a CSV file and extracts all values from the 'file_name' column.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        set: A set of filenames extracted from the CSV.
             Returns an empty set if the file is not found or is empty.
    """
    filenames = set()
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return filenames
        
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'file_name' not in reader.fieldnames:
                print(f"Error: 'file_name' column not found in {filepath}")
                return filenames
            for row in reader:
                if row['file_name']: # Ensure the filename is not empty
                    filenames.add(row['file_name'])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return filenames

def compare_files(file1_path, file2_path):
    """
    Compares two CSV files and prints filenames unique to each.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
    """
    filenames1 = get_filenames_from_csv(file1_path)
    filenames2 = get_filenames_from_csv(file2_path)

    if not filenames1 and not filenames2 and (not os.path.exists(file1_path) or not os.path.exists(file2_path)):
        # Avoid printing further if files were not found or both are empty due to errors
        return

    only_in_file1 = sorted(list(filenames1 - filenames2))
    only_in_file2 = sorted(list(filenames2 - filenames1))

    print(f"\n--- Comparison Results ---")

    if only_in_file1:
        print(f"\nFile names in '{os.path.basename(file1_path)}' but not in '{os.path.basename(file2_path)}':")
        for name in only_in_file1:
            print(f"- {name}")
    else:
        print(f"\nNo unique file names found in '{os.path.basename(file1_path)}' compared to '{os.path.basename(file2_path)}'.")

    if only_in_file2:
        print(f"\nFile names in '{os.path.basename(file2_path)}' but not in '{os.path.basename(file1_path)}':")
        for name in only_in_file2:
            print(f"- {name}")
    else:
        print(f"\nNo unique file names found in '{os.path.basename(file2_path)}' compared to '{os.path.basename(file1_path)}'.")

    if not only_in_file1 and not only_in_file2:
        if filenames1 or filenames2: # Only print this if files were actually processed
             print(f"\nBoth files contain the same set of file names or one/both were empty/invalid.")


if __name__ == "__main__":
    # --- Configuration ---
    # Please update these paths to your actual file locations
    file1_path = './extraction_results/simulation/extraction_results_model_simulation_20250525_192102.csv'
    file2_path = './extraction_results/simulation/ground_truth_simulation.csv'
    # --- End Configuration ---

    print(f"Comparing files:")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    compare_files(file1_path, file2_path) 