"""
Standalone inference script for a single image using a pre-trained 4-bit quantized model.

This script loads a 4-bit quantized DeepSeek-VL model, processes a single image with a
pre-defined prompt, and saves the generated text output to a JSON file.
"""

import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
import sys


def main():
    """
    Main function to run the standalone inference.
    """
    # --- Configuration ---
    # TODO: Update this path to your 4-bit quantized model directory.
    # This is the directory where your quantized model (e.g., created by the
    # original script's `quantize_model` function) is saved.
    MODEL_PATH = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/new_cluster_training_runs_final-ver_models/Ablation_b6-s10_short-lidar_dropout-0.15_loss/first_epoch_chkpoint/merged_model/4bit_model"

    # TODO: Update this path to the image you want to process.
    IMAGE_PATH = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/prod_test/image_20250424_105521_509112.png"
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image path does not exist: {IMAGE_PATH}")
        print("Please update the IMAGE_PATH variable in the script to a valid file.")
        sys.exit(1)

    # The prompt to use for inference.
    USER_PROMPT = "<image_placeholder> Analyze the given image from a real toy vehicle's front-facing camera (The real toy vehicle travels at half the speed of the simulated vehicle). First determine the lane the vehicle is on. Then determine if there is an obstacle directly ahead in the lane the vehicle is currently occupying or not. If an obstacle is present, determine how far away the obstacle is and decide the best movement option based on that. The lidar points are as follows: { 179.0: 1.1780 ; 180.0: 1.1960 ; 181.0: 1.3280 } Guidelines for Decision Making: First determine if the vehicle is currently on the right lane or left lane of the road. Then determine if no obstacle is detected in the current lane or if it is 'far away', then the vehicle should continue moving 'straight forward'. If an obstacle is detected in the current lane (not the other lane) and it is 'near' to the vehicle, it should 'slow cruise' cautiously until the vehicle is very close to the obstacle. Then if an obstacle is detected on the current lane and it is 'very close' to the vehicle (there is little to no road left between the vehicle and the obstacle), the vehicle should 'switch lane'. Response Format (Strict Adherence Required): Lane: {left lane | right lane | unclear}; Obstacles: {not on the same lane | far away | near | very close}; Decision: {decision}"

    # The top-level directory for saving results.
    OUTPUT_DIR = "./inference_output"
    # -------------------

    print("Starting standalone inference process...")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Image Path: {IMAGE_PATH}")

    # --- Create Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a unique subdirectory for this run
    run_output_dir = os.path.join(OUTPUT_DIR, f"inference_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Results will be saved in: {run_output_dir}")

    # --- Load Processor and Tokenizer ---
    print(f"Loading chat processor and tokenizer from {MODEL_PATH}...")
    try:
        vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer = vl_chat_processor.tokenizer
        print("Chat processor and tokenizer loaded.")
    except Exception as e:
        print(f"Error loading chat processor from {MODEL_PATH}: {e}")
        print("Please ensure the model path is correct and contains the necessary files (e.g., processor_config.json).")
        sys.exit(1)
        
    # --- Load 4-bit Quantized Model ---
    print(f"Loading 4-bit model from {MODEL_PATH} for inference...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
        vl_gpt = vl_gpt.eval()
        print("4-bit model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        print("Please ensure the model path is correct and contains a valid 4-bit quantized model.")
        sys.exit(1)

    # --- Prepare Inputs for Inference ---
    conversation = [
        {
            "role": "User",
            "content": USER_PROMPT,
            "images": [IMAGE_PATH],
        },
        {"role": "Assistant", "content": ""}
    ]

    print("Preparing inputs for the model...")
    try:
        pil_images = load_pil_images(conversation)
    except Exception as e:
        print(f"Error loading image {IMAGE_PATH}: {e}. Skipping.")
        sys.exit(1)

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)
    prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # --- Run Inference ---
    print("Generating response...")
    with torch.no_grad():
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            max_new_tokens=50,  # Increased tokens for a more detailed answer
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )

    # The output contains the full token sequence (prompt + new tokens).
    # We need to decode only the newly generated tokens.
    input_token_len = prepare_inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_token_len:]
    answer = tokenizer.decode(generated_tokens.cpu().tolist(), skip_special_tokens=True)
    
    print(f"Generated Answer: {answer.strip()}")

    # --- Save Result ---
    result = {
        "user_prompt": USER_PROMPT,
        "image_path": IMAGE_PATH,
        "model_response": answer.strip(),
    }

    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    inference_filename = f"{base_name}_inference_{timestamp}.json"
    result_path = os.path.join(run_output_dir, inference_filename)
    
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nInference complete. Result saved to: {result_path}")


if __name__ == "__main__":
    main() 