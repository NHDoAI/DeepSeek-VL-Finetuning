"""how to use: edit the correct_image_path variable below, then call the script in CLI and then enter the path to labels"""
import os
import json

def update_image_paths_in_json_files(directory_path):
    """
    Updates the image path in JSON files within a specified directory.

    The script assumes the image path to be updated is in:
    json_data['conversation'][0]['images'][0]

    The new image name will be derived from the JSON filename.
    The image path prefix "./test/real/images/" will be preserved,
    and the image extension will be ".png".
    """

    base_path = "./test/real/images/"

    updated_files_count = 0
    processed_files_count = 0
    error_files_count = 0

    print(f"Starting to process files in directory: {directory_path}\n")

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            processed_files_count += 1
            json_file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")

            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Derive the correct image filename from the JSON filename
                # e.g., from "image_abc_mirrored.json" -> "image_abc_mirrored"
                base_json_filename_without_ext = os.path.splitext(filename)[0]
                # Append .png to get the image filename, e.g., "image_abc_mirrored.png"
                correct_image_filename = base_json_filename_without_ext + ".png"
                
                # Construct the full correct image path
                # The path prefix is fixed as per your requirement
                correct_image_path =  base_path + correct_image_filename

                # Navigate to the target location and update the image path
                # Based on the structure: data['conversation'][0]['images'][0]
                if (isinstance(data, dict) and
                    "conversation" in data and
                    isinstance(data["conversation"], list) and
                    len(data["conversation"]) > 0 and
                    isinstance(data["conversation"][0], dict) and
                    "images" in data["conversation"][0] and
                    isinstance(data["conversation"][0]["images"], list) and
                    len(data["conversation"][0]["images"]) > 0):
                    
                    current_image_path = data["conversation"][0]["images"][0]
                    if current_image_path != correct_image_path:
                        data["conversation"][0]["images"][0] = correct_image_path
                        
                        # Write the updated JSON back to the file
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4) # Using indent=4 for pretty printing
                        print(f"  Successfully updated image path to: {correct_image_path}")
                        updated_files_count += 1
                    else:
                        #print(f"  Image path is already correct: {correct_image_path}")
                        pass
                else:
                    print(f"  Skipping {filename}: The JSON structure does not match the expected path "
                          f"'conversation[0].images[0]' or lists are empty.")
                    error_files_count += 1
            
            except json.JSONDecodeError:
                print(f"  Error: Could not decode JSON from {filename}.")
                error_files_count += 1
            except IOError as e:
                print(f"  Error: Could not read/write file {filename}. Details: {e}")
                error_files_count += 1
            except Exception as e:
                print(f"  An unexpected error occurred while processing {filename}. Details: {e}")
                error_files_count += 1
            print("-" * 30) # Separator for readability

    print(f"\n--- Processing Summary ---")
    print(f"Total JSON files found and processed: {processed_files_count}")
    print(f"Files successfully updated: {updated_files_count}")
    print(f"Files skipped or with errors: {error_files_count}")

if __name__ == "__main__":
    # Prompt the user for the directory path
    folder_path = input("Enter the path to the folder containing your JSON files: ")
    
    if os.path.isdir(folder_path):
        update_image_paths_in_json_files(folder_path)
    else:
        print(f"Error: The provided path '{folder_path}' is not a valid directory.") 