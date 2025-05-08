# Renames all JSON files in the specified folder by removing the pattern from filenames.

import os
import re

# Configuration - Change these values as needed
FOLDER_PATH = "./eval_output/inference_20250415_032150"  # Target folder
PATTERN_TO_REMOVE = "_inference"  # Pattern to remove from filenames

def rename_files(folder_path, pattern_to_remove):
    """
    Renames all JSON files in the specified folder by removing the pattern from filenames.
    
    Args:
        folder_path: Path to the folder containing files
        pattern_to_remove: String pattern to remove from filenames
    """
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Get all JSON files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Count for stats
    renamed_count = 0
    number_of_files = len(files)
    # Process each file
    for file in files:
        # Create new filename by removing the pattern before .json extension
        base_name = os.path.splitext(file)[0]  # Get filename without extension
        extension = os.path.splitext(file)[1]  # Get extension (.json)
        
        if base_name.endswith(pattern_to_remove):
            new_base_name = base_name[:-len(pattern_to_remove)]
            new_name = new_base_name + extension
            
            # Full paths for old and new files
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name)
            
            # Rename the file
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {file} → {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {file}: {e}")
    
    print(f"\nProcess completed: {renamed_count} files renamed.")
    print(f"Number of files: {number_of_files}")

# Execute the function
if __name__ == "__main__":
    rename_files(FOLDER_PATH, PATTERN_TO_REMOVE)