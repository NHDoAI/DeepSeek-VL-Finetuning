"""Used to modify the path of images in the json files to include a variable placeholder (folder path)

How to use:
python modify_image_paths.py /path/to/directory --variable VARIABLE_NAME --no-recursive
Arguments:
directory (required): The directory containing JSON files to process
--variable (optional): Name of the variable to use as placeholder (default: "working_dir")
--no-recursive (optional): Flag to prevent searching in subdirectories
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any


def modify_image_paths(json_file_path: str, variable_name: str = "working_dir") -> bool:
    """
    Modify image paths in the given JSON file to include a variable placeholder.
    
    Args:
        json_file_path: Path to the JSON file to modify
        variable_name: Name of the variable to use as placeholder
        
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # Check if the expected structure exists
        if "conversation" in data:
            for message in data["conversation"]:
                if "images" in message and isinstance(message["images"], list):
                    # Modify each image path
                    for i, image_path in enumerate(message["images"]):
                        # If the path starts with "./", replace with variable
                        if image_path.startswith("./"):
                            new_path = f"{variable_name}" + image_path[1:]
                            message["images"][i] = new_path
                            modified = True
                        # If the path doesn't have a variable but doesn't start with "/", 
                        # consider it relative and add the variable
                        elif not image_path.startswith("/") and not image_path.startswith("{"):
                            new_path = f"{variable_name}/{image_path}"
                            message["images"][i] = new_path
                            modified = True
        
        # Save the modified JSON if changes were made
        if modified:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
            
        return False
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return False


def process_directory(directory: str, variable_name: str = "working_dir", recursive: bool = True) -> Dict[str, int]:
    """
    Process all JSON files in the specified directory.
    
    Args:
        directory: Directory to search for JSON files
        variable_name: Name of the variable to use as placeholder
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        dict: Statistics about processing results
    """
    # Get all JSON files in the directory
    pattern = os.path.join(directory, "**/*.json" if recursive else "*.json")
    json_files = glob.glob(pattern, recursive=recursive)
    
    stats = {
        "total": len(json_files),
        "modified": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # Process each JSON file
    for json_file in json_files:
        try:
            if modify_image_paths(json_file, variable_name):
                stats["modified"] += 1
                print(f"Modified: {json_file}")
            else:
                stats["skipped"] += 1
                print(f"Skipped: {json_file} (no changes needed)")
        except Exception as e:
            stats["errors"] += 1
            print(f"Error: {json_file} - {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Modify image paths in JSON files to include a variable placeholder")
    parser.add_argument("directory", help="Directory containing JSON files to process")
    parser.add_argument("--variable", default="working_dir", help="Name of the variable to use (default: working_dir)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    
    args = parser.parse_args()
    
    print(f"Processing JSON files in: {args.directory}")
    print(f"Using variable name: {args.variable}")
    
    stats = process_directory(args.directory, args.variable, not args.no_recursive)
    
    print("\nSummary:")
    print(f"Total files processed: {stats['total']}")
    print(f"Files modified: {stats['modified']}")
    print(f"Files skipped: {stats['skipped']}")
    print(f"Errors encountered: {stats['errors']}")


if __name__ == "__main__":
    main() 