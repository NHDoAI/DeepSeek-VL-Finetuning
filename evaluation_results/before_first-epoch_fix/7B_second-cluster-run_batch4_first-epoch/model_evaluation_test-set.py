# remember to put license comment from author back

"""get the model answer on a json set. Uses the image path from json files and generate answer for that image, then save the results to a folder."""


import os
import glob
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM
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
    model_path = "/home/ubuntu/DeepSeek-VL-Finetuning/training_code/cluster_training/saved_model_first-run_batch_4_test/models/merged_model_first-epoch/"
    

    # Load the chat processor (and tokenizer) from the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the quantized model  
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto"  # Automatically assigns device placement
    )
    vl_gpt = vl_gpt.eval()

    # Get the user prompt
    user_prompt =  """<image_placeholder> Analyze the given image from the vehicle's front-facing camera. First determine if the vehicle is on the left or the right lane. Then determine if there is an obstacle directly ahead in the lane the vehicle is currently occupying or not. If an obstacle is present, determine how far away the obstacle is and decide the best movement option to keep going forward or switch to the other lane or navigate safely.
    Guidelines for Decision Making:
    1) First determine if the vehicle is currently on the right lane or left lane of the road.
    2) Then determine if no obstacle is detected in the current lane or if it is far away, then the vehicle should continue moving "straight forward".
    3) If an obstacle is detected in the current lane (not the other lane) and it is near to the vehicle, it should "slow cruise" cautiously until the vehicle is very close to the obstacle.
    4) If an obstacle is detected on the current lane and it is very close to the vehicle (there is little to no road left between the vehicle and the obstacle), the vehicle should switch to the other lane by either "change lane left" or "change lane right", following this logic:
    - If the vehicle is currently on the left lane, it should "change lane right".
    - If the vehicle is currently on the right lane, it should "change lane left".
Response Format (Strict Adherence Required):
[Scene analysis and Reasoning]
- The vehicle is currently in the {left/right} lane.
- Obstacles are {choose one from: "not on the same lane", "far away", "near", "very close"}.
- Based on these conditions, the optimal movement decision is: {decision from: "straight forward", "slow cruise", "change lane left", "change lane right"}.
[Final Decision]
{decision}"""

    # Get all image files in the folder /data/gazebo/camera/front and its subfolders (supporting common image formats)
    test_labels_dir = os.getenv('TEST_LABELS_PATH')
    test_files = [os.path.join(test_labels_dir, fname) for fname in os.listdir(test_labels_dir) if fname.endswith('.json')]

    label_data = []
    for file in test_files:
        with open(file, "r") as label_file:
            label_data.append(json.load(label_file))

    img_paths = [label_data[i]["conversation"][0]["images"][0] for i in range(len(label_data))]

    # Process each image individually
    for i, img_path in enumerate(img_paths):
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
                "content": user_prompt,
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
                    "content": user_prompt,
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
        result_path = os.path.join(output_dir, inference_filename)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        
        # Copy image to output directory
        if save_image_flag:
            try:
                new_img_path = os.path.join(output_dir, img_filename)
                shutil.copy2(img_path, new_img_path)
            except Exception as e:
                print(f"Warning: Could not copy image {img_filename}: {e}")
        
        print(f"\nProcessed: {img_filename}")
        print(f"Response: {answer}")
        print(f"Saved to: {result_path}")

    print(f"\nAll results saved in: {output_dir}")

if __name__ == "__main__":
    main() 