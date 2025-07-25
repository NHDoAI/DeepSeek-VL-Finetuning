import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from dotenv import load_dotenv

# Import specific classes from deepseek_vl
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor

# Import List from typing for Python < 3.9 compatibility
from typing import List


load_dotenv()

def merge_qlora_model_with_new_tokens(
    base_model_name: str,
    adapter_checkpoint: str,
    output_dir: str,
    new_special_tokens: List[str]
):
    """
    Merges a QLoRA adapter with new special tokens into a base model.

    The process involves:
    1. Loading the processor and tokenizer.
    2. Adding new special tokens to the tokenizer.
    3. Loading the base model with quantization.
    4. Resizing the base model's token embeddings to accommodate the new tokens.
       This assumes the model has a 'language_model' attribute whose embeddings need resizing,
       consistent with how LLaVA or similar multi-modal models are often structured and trained with PEFT.
    5. Loading the PeftModel (base + adapter).
    6. Merging the adapter into the base model.
    7. Saving the merged model and the updated processor/tokenizer.
    """
    print("Starting model merging process with new tokens...")
    print(f"Base model: {base_model_name}")
    print(f"Adapter checkpoint: {adapter_checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"New special tokens: {new_special_tokens}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load processor and tokenizer
    # Use trust_remote_code=True if your base model requires custom code.
    # For deepseek_vl, this might be handled internally or might still be needed
    # depending on its specific implementation.
    try:
        # Explicitly use VLChatProcessor from deepseek_vl
        processor = VLChatProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer
        print("Deepseek-VL VLChatProcessor and tokenizer loaded.")
    except Exception as e:
        print(f"Failed to load Deepseek-VL VLChatProcessor. Error: {e}")
        print("Ensure the base_model_name points to a valid Deepseek-VL model and the library is correctly installed.")
        # Fallback to AutoTokenizer if processor loading fails, though this might be less ideal for multi-modal models
        print("Attempting to load tokenizer directly using AutoTokenizer as a fallback.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            processor = None # No full processor loaded
            print("AutoTokenizer loaded directly (fallback).")
        except Exception as e_tok:
            print(f"Failed to load AutoTokenizer directly. Error: {e_tok}")
            return None

    # 2. Add new special tokens to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    print(f"Added {len(new_special_tokens)} new special tokens to the tokenizer.")

    # It's good practice to set pad_token if it's not already set.
    # Often, eos_token is used as pad_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ('{tokenizer.eos_token}')")
    
    # 3. Load base model configuration and update it
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id # Set if your model uses BOS and it's in tokenizer

    # Prepare quantization config for loading the base model in 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Quantization config prepared.")

    # 4. Load the base model with quantization
    print(f"Loading base model '{base_model_name}' (as Deepseek-VL MultiModalityCausalLM) with 4-bit quantization...")
    # Use MultiModalityCausalLM for loading the base model
    base_model = MultiModalityCausalLM.from_pretrained(
        base_model_name,
        config=config,
        quantization_config=quantization_config,
        trust_remote_code=True, # Keep this, as custom code might be involved
        device_map="auto" # Automatically distribute layers across available devices
    )
    print("Base model (Deepseek-VL MultiModalityCausalLM) loaded.")

    # 5. Resize the base model's token embeddings
    # This step is crucial and must match how it was done during training.
    # The training script used `model.language_model.resize_token_embeddings`.
    # This implies `base_model` here is a multi-modal model with a `language_model` attribute.
    print(f"Resizing token embeddings of the language model component to: {len(tokenizer)}")
    try:
        # Assuming the model structure has a 'language_model' attribute (e.g., LLaVA)
        # which is the actual Hugging Face LM (e.g., LlamaForCausalLM)
        if hasattr(base_model, 'language_model') and hasattr(base_model.language_model, 'resize_token_embeddings'):
            base_model.language_model.resize_token_embeddings(len(tokenizer))
            print("Token embeddings resized via 'base_model.language_model.resize_token_embeddings'.")
        elif hasattr(base_model, 'resize_token_embeddings'):
             # If base_model itself is the LM or directly supports resizing
            base_model.resize_token_embeddings(len(tokenizer))
            print("Token embeddings resized via 'base_model.resize_token_embeddings'.")
        else:
            raise AttributeError("Model does not have a recognized method for resizing token embeddings "
                                 " (checked 'language_model.resize_token_embeddings' and 'resize_token_embeddings').")
    except Exception as e:
        print(f"Error during token embedding resizing: {e}")
        print("Please ensure the model structure and resizing path are correct for your base model.")
        return None
        
    # 6. Load the PeftModel (base + adapter)
    # The `base_model` now has resized embeddings. PeftModel will load adapter weights,
    # including full weights for `modules_to_save` (like embeddings) and LoRA weights for targets.
    print(f"Loading PEFT model with adapter from '{adapter_checkpoint}'...")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    print("PEFT model loaded.")

    # 7. Merge the adapter into the base model
    print("Merging adapter weights into the base model...")
    merged_model = model_with_adapter.merge_and_unload()
    print("Adapter merged and unloaded.")

    # 8. Save the merged model and the updated processor/tokenizer
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)
    print("Merged model saved.")

    if processor:
        print(f"Saving processor (with updated tokenizer) to {output_dir}...")
        processor.save_pretrained(output_dir) # This should save the modified tokenizer as part of the processor
        print("Processor saved.")
    else:
        # If processor wasn't loaded, save tokenizer directly
        print(f"Saving tokenizer to {output_dir}...")
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    
    # Explicitly save tokenizer again (as in your original script, for robustness)
    # if processor was loaded, this might be redundant but ensures tokenizer files are present.
    if processor: 
        print(f"Explicitly saving tokenizer again to {output_dir}...")
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer explicitly saved again.")

    print(f"Successfully merged model and saved all components to {output_dir}")
    return output_dir


if __name__ == '__main__':
    # Hard-coded arguments
    base_model_name = os.getenv('MODEL_1.3B_CHKPOINT')
    adapter_checkpoint = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/cluster_training_runs/1.3b_Option-B-first-run_accuracy/best_chkpoint/adapter/" 
    output_dir = "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/cluster_training_runs/1.3b_Option-B-first-run_accuracy/best_chkpoint/merged_model_output"
    new_tokens = ["<LANE>", "<OBS>", "<DEC>"]

    merge_qlora_model_with_new_tokens(
        base_model_name=base_model_name,
        adapter_checkpoint=adapter_checkpoint,
        output_dir=output_dir,
        new_special_tokens=new_tokens
    )

    # Example usage from command line:
    # python merge_qlora_with_new_tokens.py \
    #   --base_model_name "path/to/your/base_multimodal_model_or_llm" \
    #   --adapter_checkpoint "path/to/your/adapter_checkpoint" \
    #   --output_dir "path/to/your/merged_model_output" \
    #   --new_tokens "<LANE>" "<OBS>" "<DEC>" 