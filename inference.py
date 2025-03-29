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

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# specify the path to the model
#model_path = "deepseek-ai/deepseek-vl-7b-chat"
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
prompt = """<image_placeholder>You are looking through the camera of a turtlebot3 burger robot inside a gazebo simulation environment.
 The robot is on a road. Tell me do you see anything on the road standing in front of the bot? If there is something on the road that will
 block the robot's path, that means the robot should change lane to the left or right. Based on what you see, tell me what the robot
 should do and give your reasoning."""

# Initial conversation setup
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>You are looking through the camera of a turtlebot3 burger robot inside a gazebo simulation environment. The robot is on a road. Tell me do you see anything on the road standing in front of the bot? If there is something on the road that will block the robot's path, that means the robot should change lane to the left or right. Based on what you see, tell me what the robot should do and give your reasoning.",
        "images": ["./images/gazebo_image-1.png"],
    },
    {"role": "Assistant", "content": ""},
]

# Process each turn of conversation
def get_model_response(conversation):
    # Load images and prepare inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # Get image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate response
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

    # Decode the response
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

# Get first response
first_response = get_model_response(conversation)
conversation[1]["content"] = first_response

# Add follow-up question
conversation.extend([
    {
        "role": "User",
        "content": "If there is something on the road in front of the robot, it likely means it will block the robot's path. Should the robot in this case change lane to the left or right?",
    },
    {"role": "Assistant", "content": ""},
])

# Get second response
second_response = get_model_response(conversation)
conversation[3]["content"] = second_response

# Print the full conversation
for message in conversation:
    print(f"{message['role']}: {message['content']}\n")
