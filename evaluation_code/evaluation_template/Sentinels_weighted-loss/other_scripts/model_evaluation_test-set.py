# remember to put license comment from author back

"""get the model answer on a json set, take the image path and the user prompt from those files and pass them to the model for the forward method, then save the results to a folder."""

import os
import glob
import torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoConfig
)
import json
import shutil

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

def main():

    save_image_flag = False

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./eval_output", f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Specify the path to the model
    model_path = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver/1.3b_Option-B_version2_fixed/first-run_loss/best_chkpoint/merged_model_best_chkpoint/_4bit_model"
    

    # Load the chat processor (and tokenizer) from the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Define and add new special tokens to the tokenizer
    # This ensures the tokenizer used for processing and decoding is aware of them.
    new_special_tokens = ["<LANE>", "<OBS>", "<DEC>"]
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

    if num_added_toks > 0:
        print(f"Added {num_added_toks} new special tokens to the tokenizer: {new_special_tokens}")
    else:
        print(f"The special tokens {new_special_tokens} were already present in the loaded tokenizer.")

    # Load the quantized model  
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    # Resize model's token embeddings if its vocabulary size doesn't match the tokenizer's.
    # This is essential for the model to correctly handle the new tokens.
    current_model_vocab_size = vl_gpt.language_model.get_input_embeddings().weight.size(0)
    if current_model_vocab_size != len(tokenizer):
        print(f"Resizing model token embeddings from {current_model_vocab_size} to {len(tokenizer)}.")
        vl_gpt.language_model.resize_token_embeddings(len(tokenizer))
    else:
        print("Model token embedding size already matches the tokenizer's vocabulary size.")

    vl_gpt = vl_gpt.eval()

    # Get the user prompt

    # Get all image files in the folder /data/gazebo/camera/front and its subfolders (supporting common image formats)
    test_labels_real_dir = os.getenv('TEST_LABELS_PATH_REAL')
    test_labels_sim_dir = os.getenv('TEST_LABELS_PATH_SIM')
    test_files_real = [os.path.join(test_labels_real_dir, fname) for fname in os.listdir(test_labels_real_dir) if fname.endswith('.json')]
    test_files_sim = [os.path.join(test_labels_sim_dir, fname) for fname in os.listdir(test_labels_sim_dir) if fname.endswith('.json')]

    sample_data_real = []
    for file_path in test_files_real:
        with open(file_path, "r") as label_file:
            data = json.load(label_file)
            img_path = data["conversation"][0]["images"][0]
            prompt = data["conversation"][0]["content"]
            sample_data_real.append([img_path, prompt])
    
    sample_data_sim = []
    for file_path in test_files_sim:
        with open(file_path, "r") as label_file:
            data = json.load(label_file)
            img_path = data["conversation"][0]["images"][0]
            prompt = data["conversation"][0]["content"]
            sample_data_sim.append([img_path, prompt])

    output_dir_real = os.path.join(output_dir, "REAL")
    output_dir_sim = os.path.join(output_dir, "SIM")
    os.makedirs(output_dir_real, exist_ok=True)
    os.makedirs(output_dir_sim, exist_ok=True)

    datasets_to_process = [
        ("REAL", sample_data_real, output_dir_real),
        ("SIM", sample_data_sim, output_dir_sim)
    ]

    for dataset_name, current_sample_data, current_dataset_output_dir in datasets_to_process:
        print(f"\nProcessing {dataset_name} samples...")
        if not current_sample_data:
            print(f"No samples found for {dataset_name} dataset. Skipping.")
            continue

        for i, (img_path, current_user_prompt) in enumerate(current_sample_data):
            # Extract the absolute path of the image folder
            absolute_img_dir_path = os.path.dirname(os.path.abspath(img_path))
            # Extract the image filename (without path)
            img_filename = os.path.basename(img_path)
            # Get the base name without extension
            base_name = os.path.splitext(img_filename)[0]
            
            full_img_path = os.path.join(absolute_img_dir_path, img_filename)
            
            # Create a conversation with the user prompt and single image
            conversation = [
                {
                    "role": "User",
                    "content": current_user_prompt,
                    "images": [img_path],  # Only one image at a time
                },
                {"role": "Assistant", "content": ""}
            ]

            # Load single image
            pil_images = load_pil_images(conversation)

            # Prepare inputs for the model and set to the correct device
            prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)

            # Run the image encoder to get image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # Generate the response from the model
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

            # Decode the answer
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Save result for this image with proper naming
            result = {
                "conversation": [
                    {
                        "role": "User",
                        "content": current_user_prompt,
                        "images": [img_path],
                    },
                    {
                        "role": "Assistant", 
                        "content": answer
                    }
                ]
            }
            
            # Save as JSON with image name
            inference_filename = f"{base_name}.json"
            result_path = os.path.join(current_dataset_output_dir, inference_filename)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=4)
            
            # Copy image to output directory
            if save_image_flag:
                try:
                    new_img_path = os.path.join(current_dataset_output_dir, img_filename)
                    shutil.copy2(img_path, new_img_path)
                except Exception as e:
                    print(f"Warning: Could not copy image {img_filename} for {dataset_name} dataset: {e}")
            
            print(f"\nProcessed ({dataset_name}): {img_filename}")
            print(f"Response: {answer}")
            print(f"Saved to: {result_path}")

        print(f"\nAll {dataset_name} results saved in: {current_dataset_output_dir}")

    print(f"\nAll results saved in parent directory: {output_dir}")

if __name__ == "__main__":
    main() 