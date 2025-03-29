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
    output_dir = os.path.join("../output", f"test_vision_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Specify the path to the model
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    
    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["vision_model", "aligner"]
    )

    # Load the chat processor (and tokenizer) from the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the quantized model  
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto"  # Automatically assigns device placement
    )
    vl_gpt = vl_gpt.eval()

    # Get the user prompt
    #user_prompt = "<image_placeholder>" + input("Enter your prompt regarding the images:")
    user_prompt =  """<image_placeholder> What do you see in the image?"""
    
    # user_prompt =  """<image_placeholder> Analyze the given image from the vehicle's front-facing camera by strictly following this guideline step by step: The Vehicle is on a road with two lanes. Do you see the vehicle is on the left or right lane? Are there any obstacles ahead in the lane the vehicle is currently occupying? If there is an obstacle, what is it and is it far away or close?"""

    # Get all image files in the folder /data/gazebo/camera/front and its subfolders (supporting common image formats)
    image_folder = "../data/test_images_real"
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