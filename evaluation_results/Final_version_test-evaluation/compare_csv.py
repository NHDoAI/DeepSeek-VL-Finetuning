import pandas as pd
import argparse

def compare_csv_files(file1_path, file2_path):
    """
    Compares two CSV files row by row and prints out any rows that have differences.
    If no differences are found, it prints "No discrepancies detected."
    """
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError:
        print("Error: One or both files not found. Please check the paths.")
        return

    if df1.shape != df2.shape:
        print("Warning: Files have different dimensions.")

    discrepancies_found = False
    for index, row1 in df1.iterrows():
        row2 = df2.iloc[index]
        if not row1.equals(row2):
            discrepancies_found = True
            print(f"Discrepancy in row {index + 2}:") # +2 for header and 0-indexed
            print(f"  File 1: {row1.to_dict()}")
            print(f"  File 2: {row2.to_dict()}")
    
    if not discrepancies_found:
        print("No discrepancies detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two CSV files for discrepancies.")
    parser.add_argument("file1", help="Path to the first CSV file.")
    parser.add_argument("file2", help="Path to the second CSV file.")
    args = parser.parse_args()

    compare_csv_files(args.file1, args.file2) 