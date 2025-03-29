#!/usr/bin/env python3
# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import glob
import torch
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import json
import shutil
from datetime import datetime

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Specify the path to the model
    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    

    # Load the chat processor (and tokenizer) from the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the model  
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Get the user prompt
    #user_prompt = "<image_placeholder>" + input("Enter your prompt regarding the images:")
    user_prompt =  """<image_placeholder> Analyze the given image from the vehicle's front-facing camera. First determine if the vehicle is on the left or the right lane. Then determine if there is an obstacle directly ahead in the lane the vehicle is currently occupying. If an obstacle is present, determine how far away the obstacle is and decide the best movement option to keep going forward or switch to the other lane or navigate safely.
Guidelines for Decision Making:
    • If no obstacles are detected in the current lane, the vehicle should continue moving "straight forward".
    • If an obstacle is detected in the current lane(not the other lane) but it is still far away, the vehicle should "step forward" cautiously until there is no or very little space to the obstacle.
    • If an obstacle is detected on the current lane and it is very close to the vehicle (there is no space between the vehicle and the obstacle), the vehicle should switch to the other lane by either "turn left" or "turn right" by following these steps:
        ◦ The vehicle should first determine if is currently on the right side or left side of the road.
        ◦ If the vehicle is on the left side, it should "turn right" to switch lanes if clear.
        ◦ If the vehicle is on the right side, it should "turn left" to switch lanes if clear.
Response Format (Strict Adherence Required):
[Scene Analysis]
- Detected Lane: {left/right}
- Obstacle Present: {yes/no}
- Obstacle Distance: {near/far}
- Clearance Available: {yes/no}
[Reasoning]
- The vehicle is currently in the {left/right} lane.
- An obstacle {is/is not} on the same lane and is located {near/far}.
- The available space in the adjacent lane is {sufficient/insufficient}.
- Based on these conditions, the optimal movement decision is: {decision from: "straight forward", "step forward", "step backward", "turn left", "turn right"}.
[Final Decision]
{decision}"""

    # Get all image files in the folder /data/gazebo/camera/front and its subfolders (supporting common image formats)
    image_folder = "./data/gazebo/camera/"
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

    if not image_paths:
        print(f"No images found in {image_folder} or its subfolders")
        return

    # Process each image individually
    for img_path in image_paths:
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
        
        # Get relative path from the base image folder
        rel_path = os.path.relpath(os.path.dirname(img_path), image_folder)
        
        # Create the same subfolder structure in output directory
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Save files in the corresponding subfolder
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        
        # Copy image to output directory maintaining folder structure
        new_img_path = os.path.join(output_subdir, img_filename)
        shutil.copy2(img_path, new_img_path)
        
        # Save result for this image in the same subfolder
        result = {
            "prompt": user_prompt,
            "response": answer,
            "image": img_filename,
            "relative_path": rel_path
        }
        
        # Save as JSON in the same subfolder
        result_path = os.path.join(output_subdir, f"{base_name}_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        
        print(f"\nProcessed: {img_filename}")
        print(f"Response: {answer}")

    print(f"\nAll results saved in: {output_dir}")

if __name__ == "__main__":
    main() 