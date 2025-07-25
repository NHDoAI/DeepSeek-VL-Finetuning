import os
import torch
import json
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# Load environment variables from a .env file
load_dotenv()

def run_single_inference(model_name_or_path: str, input_json_path: str, output_dir: str):
    """
    Runs inference on a single data sample using a specified DeepSeek-VL model.

    Args:
        model_name_or_path (str): The path to the pre-trained model checkpoint.
        input_json_path (str): The path to the input JSON file containing the conversation.
        output_dir (str): The directory to save the output JSON file.
    """
    print("Starting single inference...")
    print(f"  Model: {model_name_or_path}")
    print(f"  Input JSON: {input_json_path}")

    # --- 1. Load Model and Processor ---
    print("\nLoading model and processor...")
    try:
        vl_chat_processor = VLChatProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = vl_chat_processor.tokenizer
        
        # Define 4-bit quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Explicitly set use_gradient_checkpointing to False for inference
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.use_gradient_checkpointing = False

        print("Loading 4-bit quantized model...")
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
        ).eval()
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Please ensure the model path is correct and you have the necessary dependencies installed.")
        return
        
    print("Model and processor loaded successfully.")

    # --- 2. Load and Prepare Data Sample ---
    print(f"\nLoading data from {input_json_path}...")
    try:
        with open(input_json_path, "r") as f:
            data = json.load(f)
        
        # Extract only the user's part of the conversation for inference
        user_turn = data["conversation"][0]
        user_prompt = user_turn["content"]
        img_path_from_json = user_turn["images"][0]

        # Construct the conversation for inference, with an empty assistant response
        inference_conversation = [
            user_turn,
            {"role": "Assistant", "content": ""}
        ]
        
    except (KeyError, IndexError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing JSON file '{input_json_path}': {e}.")
        print("Please ensure it is a valid path and contains the expected 'conversation' format.")
        return

    # --- 3. Run Inference ---
    print(f"Running inference on image: {img_path_from_json}")
    
    try:
        # Pass the correctly formatted conversation to the helpers
        pil_images = load_pil_images(inference_conversation)
        
        # Prepare inputs for the model
        prepare_inputs = vl_chat_processor(
            conversations=inference_conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)
        
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        print("Starting generation...")
        start_time = time.monotonic()
        with torch.no_grad():
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id
            )
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"Peak GPU memory load during inference: {peak_memory:.2f} MiB")
        end_time = time.monotonic()
        inference_duration = end_time - start_time
        print(f"Inference generation took {inference_duration:.4f} seconds.")

        # Decode the full output and then extract only the assistant's response
        full_response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # The prompt is constructed as f"User: {prompt}\n\nAssistant: "
        # We find the start of the assistant's part to isolate its actual response.
        assistant_prompt_marker = "Assistant:"
        assistant_response_start = full_response.rfind(assistant_prompt_marker)
        if assistant_response_start != -1:
            answer = full_response[assistant_response_start + len(assistant_prompt_marker):].strip()
        else:
            # Fallback if the marker isn't found (should not happen with this model)
            answer = full_response.strip()

        print(f"\nGenerated response: {answer}")

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return

    # --- 4. Save the Output ---
    output_conversation = [
        {
            "role": "User",
            "content": user_prompt,
            "images": [img_path_from_json],
        },
        {
            "role": "Assistant",
            "content": answer
        }
    ]
    
    result = {"conversation": output_conversation}

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    output_filename = f"{base_name}_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
        
    print(f"\nInference complete. Result saved to: {output_path}")


def main():
    """Main function to parse arguments and run the inference script."""
    parser = argparse.ArgumentParser(description="Run single inference with a DeepSeek-VL model.")
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        choices=['1.3b', '7b'],
        help="The model version to use ('1.3b' or '7b')."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the single input JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./single_inference_output",
        help="Directory to save the output JSON file. (Default: ./single_inference_output)"
    )
    args = parser.parse_args()

    # Get model checkpoint path from environment variables
    if args.model_version == '1.3b':
        model_path = os.getenv('MODEL_1.3B_CHKPOINT')
        if not model_path:
            raise ValueError("Environment variable 'MODEL_1.3B_CHKPOINT' is not set.")
    elif args.model_version == '7b':
        model_path = os.getenv('MODEL_7B_CHKPOINT')
        if not model_path:
            raise ValueError("Environment variable 'MODEL_7B_CHKPOINT' is not set.")
    else:
        # This case is handled by argparse `choices`
        raise ValueError("Invalid model version specified.")

    run_single_inference(model_path, args.input_json, args.output_dir)

if __name__ == "__main__":
    main() 