"""merge adapter weights into the base model"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from peft import PeftModel, PeftConfig
import os
import torch
from dotenv import load_dotenv 

load_dotenv()
# Step 1: Load base model (same one you used for fine-tuning)
base_model_name = os.getenv('MODEL_1.3B_CHKPOINT')  # or your original base
print(base_model_name)

vl_chat_processor = VLChatProcessor.from_pretrained(base_model_name)
tokenizer = vl_chat_processor.tokenizer

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Note: llm_int8_skip_modules is ignored in 4-bit mode.
    )

config = AutoConfig.from_pretrained(base_model_name)
# --- Set definition of special tokens ---
config.pad_token_id = tokenizer.eos_token_id
config.eos_token_id = tokenizer.eos_token_id
config.bos_token_id = tokenizer.bos_token_id


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    quantization_config=quantization_config,
    config=config,
    device_map="auto"
)

# Step 2: Load the adapter from your fine-tuning checkpoint
adapter_checkpoint = "./experimental_1.3b_less-lora/models/later_save/adapter/"
model_with_adapter = PeftModel.from_pretrained(base_model, adapter_checkpoint)

# Step 3: Merge adapter weights into the base model
merged_model = model_with_adapter.merge_and_unload()

# Step 4: Save the fully merged model for deployment
output_dir = "./experimental_1.3b_less-lora/models/merged_model_later_save"


merged_model.save_pretrained(output_dir)

# Optional: Save tokenizer too
vl_chat_processor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved at {output_dir}")
