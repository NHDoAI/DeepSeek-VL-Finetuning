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

import torch
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
#model_path = "deepseek-ai/deepseek-vl-1.3b-chat"

# Create quantization config
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["vision_model", "aligner"]
    #bnb_4bit_compute_dtype=torch.float16
)

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Modified model loading with quantization
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto"  # This handles device placement automatically
)

# No need for manual dtype conversion or .cuda() call when using quantization
vl_gpt = vl_gpt.eval()

# single image conversation example
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>You are looking through the camera of a turtlebot3 burger robot inside a gazebo simulation environment. The robot is on a road. If there is something on the road that will block the robot's path, that means the robot should change lane to the left if it is currently on the right lane, or to the right if it is currently on the left lane. Is the robot currently on the right lane or the left lane and do you see anything on the road that will block the robot's path? If yes to which lane should the robot change?",
        "images": ["./images/gazebo_image-1.png"],
    },
    {"role": "Assistant", "content": ""},

]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
# prepare_inputs = vl_chat_processor(
#     conversations=conversation, images=pil_images, force_batchify=True
# ).to(vl_gpt.device)

prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)

prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
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

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
