import json
import os
import random
from dotenv import load_dotenv
import time
import subprocess

import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from peft import PeftModel
from deepseek_vl.utils.io import load_pil_images
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR

import numpy as np

import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cluster_flag = True
model_type = "1.3B"
eval_mode = "loss" # "loss" or "accuracy"

# ------ Configuration & Hyperparameters ------
hyperparameters = {
    "base_checkpoint_dir": "./continued_training_checkpoints/",
    "run_name_prefix": "1.3b_continued-train/v1",
    "adapter_checkpoint_to_load": "./1.3b_final-train/first_version/12-06-2025_loss/best_chkpoint",
    "eval_every_n_steps": 100,
    "master_seed": 42,
    "num_workers_dataloader": 4,
    "batch_size": 6,
    "max_epochs": 10,
    "learning_rate": 5e-5,
    "min_lr": 1e-6,
    "max_lr_reductions": 4,
    "early_stopping_enabled": True,
    "early_stopping_patience": 3,
    "wandb_project_name": "deepseek-vl-continued-training",
    "real_image_eval_weight": 0.8,
    "sim_image_eval_weight": 0.2,

    "obstacle_category_weights": {
        "not on the same lane": 0.5,
        "far away": 1.0,
        "near": 1.5,
        "very close": 2.0
    },

    # Scheduler settings
    "lr_scheduler_type": "CosineAnnealingWarmRestarts", # "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "StepLR"
    
    # Scheduler-specific params
    "cosine_warm_restarts_t_0": 100,
    "cosine_warm_restarts_t_mult": 1,
    "step_lr_step_size": 100,
    "step_lr_gamma": 0.75,
    "lr_scheduler_patience": 1, # For ReduceLROnPlateau
    "lr_scheduler_factor": 0.5, # For ReduceLROnPlateau

    "manual_checkpoint_steps": [],
    "manual_eval_save": True,
    
    "improvement_delta": 0.0001,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip_norm": 1.0,
    
    "warmup_steps": 100,
}

# Mode-dependent parameters using inline conditionals
hyperparameters["lr_scheduler_mode"] = "max" if eval_mode == "accuracy" else "min"
hyperparameters["improvement_delta"] = 0.001 if eval_mode == "accuracy" else 0.0001


# Construct wandb_run_name, checkpoint_dir and best_checkpoint_path using other hyperparameters
hyperparameters["wandb_run_name"] = f"{hyperparameters['run_name_prefix']}_{eval_mode}"
hyperparameters["checkpoint_dir"] = os.path.join(
    hyperparameters["base_checkpoint_dir"],
    f"{hyperparameters['run_name_prefix']}_{eval_mode}/"
)


# ------ Define useful functions and classes ------

# --- Function to log memory usage ---

def get_gpu_memory_info(gpu_id):
    """
    Uses nvidia-smi to query memory details for a specific GPU.
    The `-i` option specifies the GPU index.
    """
    try:
        # Query memory info for the specified GPU.
        # The '--format=csv,noheader,nounits' flag provides a clean CSV without extra headers or units.
        cmd = [
            'nvidia-smi',
            '-i', str(gpu_id),
            '--query-gpu=memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Expecting one line of CSV output for memory info, e.g., "11178, 230, 10948"
        line = result.stdout.strip()
        total, used, free = [item.strip() for item in line.split(',')]
        return {
            'Memory Total (MiB)': total,
            'Memory Used (MiB)': used,
            'Memory Free (MiB)': free
        }
    except subprocess.CalledProcessError as e:
        print("Error executing nvidia-smi:", e.stderr)
        return None
    except Exception as e:
        print("Unexpected error:", e)
        return None

def log_memory_cluster(prefix=""):
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_memory = properties.total_memory / (1024**2)  # Total memory in bytes
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**2) # Reserved memory in bytes
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2) # Allocated memory in bytes
    #free_memory = reserved_memory - allocated_memory  # Free memory in reserved space
    # allocated = torch.cuda.memory_allocated() / (1024**2)  # In MB
    # reserved = torch.cuda.memory_reserved() / (1024**2)    # In MB
    print(prefix)
    print(f"{prefix} GPU Memory - Total: {total_memory:.2f} MB | Allocated: {allocated_memory:.2f} MB | Reserved: {reserved_memory:.2f} MB") #| Free: {free_memory:.2f} MB

def log_memory_flex(cluster_flag=False, message="GPU Memory:", gpu_id=0, device_name="GPU"):
    if not cluster_flag:
        print(f"CUDA Device Name: {device_name}")
        gpu_memory = get_gpu_memory_info(gpu_id)
        print(f"PyTorch is using GPU {gpu_id}:")
        print(gpu_memory)
    else:
        log_memory_cluster(message)

# --- Function to split data files into training and evaluation sets, currently not used ---
def split_data_files(data_dir, eval_ratio=0.1, seed=42):
    """
    Split the data files into training and evaluation sets.
    
    Args:
        data_dir (str): Directory containing the JSON data files.
        eval_ratio (float): Proportion of data to use for evaluation.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: Lists of file paths for training and evaluation.
    """
    random.seed(seed)
    
    data_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.json')]
    random.shuffle(data_files)
    
    split_idx = int(len(data_files) * (1 - eval_ratio))
    train_files = data_files[:split_idx]
    eval_files = data_files[split_idx:]
    
    print(f"Dataset split: {len(train_files)} training files, {len(eval_files)} evaluation files")
    return train_files, eval_files

# --- Custom dataset class for loading and processing data ---
class MyMultimodalDataset(Dataset):
    def __init__(self, data_files, chat_processor):
        """
        Args:
            data_files (list): List of paths to JSON files containing examples.
            chat_processor (VLChatProcessor): An instance that handles image and text processing.
        """
        self.chat_processor = chat_processor
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        json_data_path = self.data_files[idx]

        with open(json_data_path, "r") as label_file:
            label_data = json.load(label_file)
        
        conversation = label_data["conversation"]

        categories = label_data.get("categories", {})    

        pil_images = load_pil_images(conversation)

        inputs = self.chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True  # Ensures outputs are in a batch-ready format.
        )
        return {"model_inputs": inputs, "categories": categories}
    
# --- Custom collate function for padding and concatenating tensors ---
def custom_collate_fn(batch, pad_keys=None, pad_values=None, tokenizer=None):
    """
    Collate a list of samples into a batch.
    Each sample in the input 'batch' is expected to be a dictionary:
    {"model_inputs": chat_processor_output_dict_or_object, "categories": categories_dict}
    
    Args:
        batch (list): List of sample dictionaries from MyMultimodalDataset.
        pad_keys (list, optional): List of keys within "model_inputs" that need padding.
        pad_values (dict, optional): A dictionary mapping keys in pad_keys to their padding values.
                                     If None, defaults are constructed.
        tokenizer: Tokenizer instance, used for default padding value of "input_ids".
    
    Returns:
        dict: A batch dictionary. Keys from "model_inputs" will be collated tensors.
              An additional key "categories" will hold a list of category dictionaries.
    """
    if not batch:
        return {}

    # 1. Separate model_inputs and categories
    model_inputs_list = [item["model_inputs"] for item in batch] # List of BatchedVLChatProcessorOutput objects
    categories_list = [item["categories"] for item in batch]

    if not model_inputs_list: # Should not happen if batch is not empty and dataset is consistent
        if categories_list:
             return {"categories": categories_list}
        return {}

    # 2. Setup pad_keys and the actual padding values map
    if pad_keys is None:
        # "labels" is often needed if your model produces variable length sequences for targets
        pad_keys = ["input_ids", "attention_mask", "images_seq_mask", "labels"] 

    # This map will hold the value to use for padding for each key in pad_keys
    actual_pad_values_map = {}
    # First, populate with defaults for all keys that might be padded
    for pk_candidate in ["input_ids", "attention_mask", "images_seq_mask", "labels"] + (pad_keys if pad_keys else []):
        if pk_candidate == "input_ids":
            pad_val_default = 0 # Fallback if no tokenizer or specific ids
            if tokenizer:
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    pad_val_default = tokenizer.pad_token_id
                elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None: # Fallback to EOS if PAD is not set
                    pad_val_default = tokenizer.eos_token_id
            actual_pad_values_map[pk_candidate] = pad_val_default
        elif pk_candidate == "labels":
            actual_pad_values_map[pk_candidate] = -100
        elif pk_candidate in ["attention_mask", "images_seq_mask"]:
            actual_pad_values_map[pk_candidate] = 0
        else: # Default for any other key potentially in pad_keys
             if pk_candidate not in actual_pad_values_map: # Avoid overwriting specific defaults if pk_candidate was already handled
                actual_pad_values_map[pk_candidate] = 0
    
    # If user provided a pad_values dict, use it to update/override the defaults
    if pad_values is not None:
        actual_pad_values_map.update(pad_values)

    # 3. Collate the model_inputs_list
    collated_model_inputs = {}
    first_model_input_sample = model_inputs_list[0] # This is a BatchedVLChatProcessorOutput object
    
    # Iterate over keys present in the first sample's model inputs
    # Assumes BatchedVLChatProcessorOutput supports .keys()
    for key in first_model_input_sample.keys():
        # Use dictionary-style access (obj[key]) for BatchedVLChatProcessorOutput
        # Assumes BatchedVLChatProcessorOutput supports __getitem__
        is_tensor_in_first = isinstance(first_model_input_sample[key], torch.Tensor)

        if key in pad_keys and is_tensor_in_first:
            # .get() is fine for actual_pad_values_map as it's a Python dict
            current_pad_value = actual_pad_values_map.get(key, 0) 

            seq_lengths = []
            for sample_model_input in model_inputs_list:
                # Accessing BatchedVLChatProcessorOutput item by key
                tensor_val = sample_model_input[key] 
                if isinstance(tensor_val, torch.Tensor) and tensor_val.ndim > 1 : # Ensure it's a tensor and has a sequence dimension
                    seq_lengths.append(tensor_val.shape[1])

            max_len = max(seq_lengths) if seq_lengths else 0
            
            padded_tensors_for_key = []
            for sample_model_input in model_inputs_list:
                tensor = sample_model_input[key] # Accessing item
                
                if not isinstance(tensor, torch.Tensor):
                    # This case should ideally not be hit if is_tensor_in_first was true and data is homogeneous
                    print(f"Warning: Expected tensor for key '{key}' in sample but got {type(tensor)}. Skipping this item for key '{key}'.")
                    continue # Skip this problematic item for this key

                # Assuming force_batchify=True made tensors [1, seq_len] or [1, N_img, ...]
                current_seq_len = tensor.shape[1] if tensor.ndim > 1 else 0
                
                pad_size = max_len - current_seq_len
                
                if pad_size > 0:
                    padding_shape = list(tensor.shape) 
                    padding_shape[1] = pad_size       
                    pad_tensor = torch.full(padding_shape, current_pad_value, dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, pad_tensor], dim=1)
                elif pad_size < 0: # Should not happen if max_len is calculated correctly
                    raise ValueError(f"pad_size is negative for key {key}. max_len: {max_len}, current_seq_len: {current_seq_len}, tensor_shape: {tensor.shape}")

                padded_tensors_for_key.append(tensor)
            
            if padded_tensors_for_key:
                try:
                    collated_model_inputs[key] = torch.cat(padded_tensors_for_key, dim=0)
                except RuntimeError as e:
                    print(f"Error during torch.cat for padded key '{key}': {e}. Shapes: {[t.shape for t in padded_tensors_for_key]}")
                    collated_model_inputs[key] = padded_tensors_for_key # Fallback to list
            # else: no valid tensors were processed for this key

        elif is_tensor_in_first:
            # For tensor keys that do NOT require padding
            # This assumes all tensors for this key across samples can be directly concatenated (dim 0).
            # `force_batchify=True` makes each `sample[key]` have a leading batch_dim of 1.
            try:
                # Gather all tensors for this key. Assuming homogeneity.
                tensors_to_cat = [s_input[key] for s_input in model_inputs_list if isinstance(s_input[key], torch.Tensor)]
                if tensors_to_cat:
                    collated_model_inputs[key] = torch.cat(tensors_to_cat, dim=0)
            except RuntimeError as e:
                problematic_shapes = [s_input[key].shape for s_input in model_inputs_list if isinstance(s_input[key], torch.Tensor)]
                print(f"Warning: Could not torch.cat tensors for key '{key}'. Storing as list of tensors. Error: {e}. Shapes: {problematic_shapes}")
                collated_model_inputs[key] = [s_input[key] for s_input in model_inputs_list if isinstance(s_input[key], torch.Tensor)] # Fallback
        else:
            # For non-tensor keys within model_inputs (if any, e.g., metadata strings)
            collated_model_inputs[key] = [s_input[key] for s_input in model_inputs_list]

    # 4. Construct the final batch dictionary
    final_batch = {}
    # Add all collated model input tensors/lists to the top level of final_batch
    final_batch.update(collated_model_inputs) 
    final_batch["categories"] = categories_list  # Add the list of category dictionaries
    
    return final_batch


def custom_forward(batch, model):
    current_batch_size = batch["input_ids"].shape[0] # current_batch_size is the number of samples in the batch, stored on CPU
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if 'pixel_values' in batch:
        batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
    inputs_embeds = model.prepare_inputs_embeds(**batch)

    del batch["pixel_values"]
    labels = batch["input_ids"].clone()
    # --- Dynamic masking based on "Assistant:" ---
    # Tokenizer should be in the global scope or passed to this function
    # Assuming `tokenizer` is the global VLChatProcessor's tokenizer
    assistant_phrase = "Assistant:"
    # Encode the phrase to get token IDs, without adding special BOS/EOS tokens for the phrase itself
    assistant_token_ids = tokenizer.encode(assistant_phrase, add_special_tokens=False)
    assistant_token_ids_tensor = torch.tensor(assistant_token_ids, device=model.device, dtype=torch.long)
    
    L_assistant = len(assistant_token_ids)
    current_input_ids = batch["input_ids"] # These are already on model.device

    for i in range(current_input_ids.shape[0]): # Iterate over batch samples
        sample_input_ids = current_input_ids[i] # Current sample's input_ids
        found_match_for_sample = False
        # Search for the "Assistant:" token sequence in the current sample
        for k_start_index in range(sample_input_ids.size(0) - L_assistant + 1):
            window = sample_input_ids[k_start_index : k_start_index + L_assistant]
            if torch.equal(window, assistant_token_ids_tensor):
                # Found "Assistant:". Mask labels up to the end of this phrase.
                # The first token of the actual model output starts *after* "Assistant:".
                # So, indices from 0 to (k_start_index + L_assistant - 1) are masked.
                mask_end_exclusive_index = k_start_index + L_assistant
                labels[i, :mask_end_exclusive_index] = -100
                found_match_for_sample = True
                break # Stop search for this sample once "Assistant:" is found
        
        if not found_match_for_sample:
            # This case should ideally not be hit if "Assistant:" is always present
            # and unique as per the problem description.
            error_msg = (f"ERROR: The 'Assistant:' phrase (tokens: {assistant_token_ids}) was not found in sample {i} of the batch. "
                      f"Input IDs for sample: {sample_input_ids.tolist()}. "
                      f"This is a critical error - the Assistant: token must be present in all samples.")
            raise ValueError(error_msg)
            
    # Original masking for padding tokens (tokens where attention_mask is 0)
    # This should be applied after the prompt masking.
    labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
    #batch["labels"] = labels # Not strictly necessary to add to batch if labels is used directly


    # Forward pass through language model
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch['attention_mask'],
        use_cache=False,
        #labels= batch["labels"]  
    )
    return outputs, current_batch_size, labels

# --- Function to evaluate the model's performance ---

def evaluate_model(model, data_dataloader, obstacle_weights):
    """
    Evaluate the model on the evaluation dataset, with loss and accuracy weighted by obstacle category.
    
    Args:
        model: The model to evaluate.
        data_dataloader: DataLoader for the evaluation dataset.
        obstacle_weights (dict): A dictionary mapping obstacle categories to weights.
        
    Returns:
        tuple: Weighted average evaluation loss and accuracy.
    """
    model.eval()

    # Initialize storage for metrics per category
    categories = list(obstacle_weights.keys())
    category_losses = {cat: 0.0 for cat in categories}
    category_samples = {cat: 0 for cat in categories}
    category_correct_tokens = {cat: 0 for cat in categories}
    category_total_tokens = {cat: 0 for cat in categories}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_dataloader):
            outputs, current_batch_size, labels = custom_forward(batch, model)
            logits  = outputs.logits 

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # --- Per-sample loss calculation ---
            # Use reduction='none' to get loss per token, then average per sample
            loss_fct_none = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            per_token_loss = loss_fct_none(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(current_batch_size, -1)
            
            valid_mask = (shift_labels != -100)
            num_valid_tokens_per_sample = valid_mask.sum(dim=1).float()
            # Handle cases with no valid tokens to avoid division by zero
            num_valid_tokens_per_sample = torch.where(num_valid_tokens_per_sample == 0, 1.0, num_valid_tokens_per_sample)
            
            sample_losses = per_token_loss.sum(dim=1) / num_valid_tokens_per_sample

            # --- Per-sample accuracy calculation ---
            preds_acc = shift_logits.argmax(-1)
            correct_tokens_mask = (preds_acc == shift_labels) & valid_mask
            
            num_correct_per_sample = correct_tokens_mask.sum(dim=1)
            # num_total_per_sample is num_valid_tokens_per_sample, but as integer
            num_total_per_sample = valid_mask.sum(dim=1)

            # --- Aggregate stats by category ---
            batch_categories = batch["categories"]
            for i in range(current_batch_size):
                obstacle_cat = batch_categories[i].get("obstacle", None)
                if obstacle_cat and obstacle_cat in categories:
                    category_losses[obstacle_cat] += sample_losses[i].item()
                    category_samples[obstacle_cat] += 1
                    category_correct_tokens[obstacle_cat] += num_correct_per_sample[i].item()
                    category_total_tokens[obstacle_cat] += num_total_per_sample[i].item()

            del batch
            del outputs
            del logits
            del shift_logits
            del shift_labels
            del per_token_loss
            del sample_losses
            del preds_acc
            del valid_mask
            del correct_tokens_mask

            torch.cuda.empty_cache()

    # --- Calculate average metrics per category ---
    avg_category_losses = {
        cat: category_losses[cat] / category_samples[cat] if category_samples[cat] > 0 else 0
        for cat in categories
    }
    avg_category_accuracies = {
        cat: category_correct_tokens[cat] / category_total_tokens[cat] if category_total_tokens[cat] > 0 else 0
        for cat in categories
    }
    
    # --- Calculate final weighted average loss and accuracy ---
    total_weight = sum(obstacle_weights[cat] for cat in categories if category_samples[cat] > 0)
    
    if total_weight > 0:
        weighted_avg_loss = sum(avg_category_losses[cat] * obstacle_weights[cat] for cat in categories) / total_weight
        weighted_avg_acc = sum(avg_category_accuracies[cat] * obstacle_weights[cat] for cat in categories) / total_weight
    else:
        # Fallback to simple average if no categories were found or total_weight is 0
        # This can happen if an eval set doesn't contain any of the specified categories.
        total_samples = sum(category_samples.values())
        total_loss = sum(category_losses.values())
        total_correct = sum(category_correct_tokens.values())
        total_tokens = sum(category_total_tokens.values())
        weighted_avg_loss = total_loss / total_samples if total_samples > 0 else 0
        weighted_avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    # Optional: print per-category stats for clarity
    print("--- Evaluation Stats by Obstacle Category ---")
    for cat in categories:
        print(f"  Category '{cat}': "
              f"Samples={category_samples[cat]}, "
              f"Avg Loss={avg_category_losses[cat]:.4f}, "
              f"Avg Acc={avg_category_accuracies[cat]:.4f}")
    print("---------------------------------------------")

    model.train()
    return weighted_avg_loss, weighted_avg_acc



# Global variables to store the best metric value and step for early stopping
best_metric_value_global = None # Will be initialized based on mode
best_step_global = -1

def initialize_global_best_metric(mode):
    """Initializes the global best metric based on the mode."""
    global best_metric_value_global
    if mode == 'accuracy':
        best_metric_value_global = float('-inf')
    elif mode == 'loss':
        best_metric_value_global = float('inf')
    else:
        raise ValueError(f"Invalid mode for global best metric: {mode}. Choose 'accuracy' or 'loss'.")

def save_checkpoint_if_improved(current_metric_value, model, step, epoch, checkpoint_path, mode, delta):
    """
    Checks if the current metric is an improvement over the global best metric and saves a checkpoint.

    Args:
        current_metric_value (float): The metric value from the current evaluation.
        model: The model to save.
        step (int): The current global step.
        epoch (int): The current epoch.
        checkpoint_path (str): The path to save the checkpoint.
        mode (str): 'min' or 'max', indicating if lower or higher is better.
        delta (float): The minimum change to qualify as an improvement.

    Returns:
        bool: True if the metric improved and a checkpoint was saved, False otherwise.
    """
    global best_metric_value_global, best_step_global

    if best_metric_value_global is None:
        # This should ideally be called once before the training loop,
        # but as a fallback if not.
        initialize_global_best_metric(mode)

    improved = False
    if mode == 'loss':
        if current_metric_value < best_metric_value_global - delta:
            best_metric_value_global = current_metric_value
            improved = True
    elif mode == 'accuracy':
        if current_metric_value > best_metric_value_global + delta:
            best_metric_value_global = current_metric_value
            improved = True
    
    if improved:
        model.save_pretrained(checkpoint_path)
        best_step_global = step  # Update global best step
        
        metadata = {
            "step": step,
            "epoch": epoch,
            "best_step": best_step_global,
            f"best_metric_value ({mode})": best_metric_value_global,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(checkpoint_path, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        wandb.log({"best_model_step": best_step_global, f"best_model_metric ({mode})": best_metric_value_global})
        print(f"Checkpoint saved to {checkpoint_path} at step {best_step_global} (epoch {epoch}) with {mode} metric: {best_metric_value_global:.4f} (Global Best)")
    return improved

# --- Class for early stopping ---
class EarlyStopping:
    def __init__(self, patience=3): # Delta, checkpoint_path, and mode are removed
        self.patience = patience
        self.counter = 0
        self.patience_met = False

    # save_checkpoint method is removed

    def check(self, metric_improved_and_saved: bool):
        """
        Updates the counter based on whether an improvement led to a checkpoint save.
        Sets patience_met flag if patience is exceeded.

        Args:
            metric_improved_and_saved (bool): True if save_checkpoint_if_improved saved a model.
        """
        if metric_improved_and_saved:
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.patience_met = True

class LRReductionMonitor:
    def __init__(self):
        self.lr_reduction_count = 0
    def compare_lr(self, old_lr, new_lr, global_step, epoch):
        if new_lr < old_lr: # Compare with LR before stepping
            print(f"Learning rate reduced from {old_lr} to {new_lr} at step {global_step} by ReduceLROnPlateau based on {eval_mode} metric.")
            self.lr_reduction_count += 1
            wandb.log({"lr_reduction_count": self.lr_reduction_count, "step": global_step, "epoch": epoch + 1})
        elif new_lr == old_lr:
                print("Current scheduler is ReduceLROnPlateau and LR didn't change.")
    

# 1) pick ONE master seed for the whole run
MASTER_SEED = hyperparameters["master_seed"]          # change this between experiments

# 2) seed every library that owns an RNG
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # (makes cuDNN deterministic; slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(MASTER_SEED)

g = torch.Generator()
g.manual_seed(MASTER_SEED)

def worker_init_fn(worker_id):
    worker_seed = MASTER_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------ Load/initialize objects ------

# --- Initialize wandb ---
run = wandb.init(
    project=hyperparameters["wandb_project_name"],  # name of your project in wandb
    name=hyperparameters["wandb_run_name"], # Constructing a more dynamic run name
    mode="online"
)

# --- Load environment variables ---
load_dotenv(override=True)

# --- Get the name of the GPU ---
gpu_id = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(gpu_id)  # 0 is the device index
print(f"PyTorch is using GPU {gpu_id}: {device_name}")

log_memory_flex(cluster_flag, message="Before original model load:",gpu_id=gpu_id, device_name=device_name)

# --- Load model related objects ---

# --- model path ---
if model_type == "7B":
    model_path = os.getenv('MODEL_7B_CHKPOINT')
elif model_type == "1.3B":
    model_path = os.getenv('MODEL_1.3B_CHKPOINT')

# --- Load chat processor and tokenizer from model path ---
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# --- Define the Quantization Configuration ---

# --- 4 bit version ---
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# --- Auto load config of the model ---
config = AutoConfig.from_pretrained(model_path)
# --- Set definition of special tokens ---
config.pad_token_id = tokenizer.eos_token_id
config.eos_token_id = tokenizer.eos_token_id
config.bos_token_id = tokenizer.bos_token_id

# --- Create quantized model based on configs ---
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=quantization_config,
    config=config,
    device_map="auto"
)

# ---- MONKEY PATCH START ----
# This is necessary because the MultiModalityCausalLM class does not
# implement get_input_embeddings and get_output_embeddings, which PEFT
# needs when saving the model.
import types

def get_input_embeddings_direct_access(self_mm_model):
    if hasattr(self_mm_model, 'language_model'):
        lang_model = self_mm_model.language_model
        if hasattr(lang_model, 'model') and hasattr(lang_model.model, 'embed_tokens'):
            return lang_model.model.embed_tokens
        elif hasattr(lang_model, 'embed_tokens'):
            return lang_model.embed_tokens
    raise AttributeError(
        "MultiModalityCausalLM's language_model does not have the expected attribute structure "
        "for input embeddings (e.g., language_model.model.embed_tokens or language_model.embed_tokens)."
    )

model.get_input_embeddings = types.MethodType(get_input_embeddings_direct_access, model)
print("Monkey-patched get_input_embeddings onto the MultiModalityCausalLM instance.")

def get_output_embeddings_direct_access(self_mm_model):
    if hasattr(self_mm_model, 'language_model') and hasattr(self_mm_model.language_model, 'lm_head'):
        return self_mm_model.language_model.lm_head
    elif hasattr(self_mm_model, 'language_model') and hasattr(self_mm_model.language_model, 'get_output_embeddings'):
        try:
            output_embeds = self_mm_model.language_model.get_output_embeddings()
            if output_embeds is not None:
                print("Warning: Used language_model.get_output_embeddings() as fallback for monkey-patch.")
                return output_embeds
        except NotImplementedError:
            pass
    raise AttributeError(
        "MultiModalityCausalLM's language_model does not have the expected lm_head attribute "
        "or a working get_output_embeddings method for output embeddings."
    )

model.get_output_embeddings = types.MethodType(get_output_embeddings_direct_access, model)
print("Monkey-patched get_output_embeddings onto the MultiModalityCausalLM instance.")
# ---- MONKEY PATCH END ----

# --- Load LoRA Adapters from checkpoint ---
print(f"Loading adapters from: {hyperparameters['adapter_checkpoint_to_load']}")
model = PeftModel.from_pretrained(model, hyperparameters["adapter_checkpoint_to_load"], is_trainable=True)
print("LoRA adapters loaded and set to trainable.")
model.print_trainable_parameters()


# --- Define checkpoint path for saving the model-checkpoints ---
best_checkpoint_path = os.path.join(hyperparameters["checkpoint_dir"], "best_chkpoint")

# --- Prepare Training Loop ---

# --- Prepare Datasets ---

train_labels_dir = os.getenv('TRAIN_LABELS_PATH')
print(train_labels_dir)

eval_labels_dir_real = os.getenv('EVAL_LABELS_PATH_REAL')
print(f"Eval Real Labels Path: {eval_labels_dir_real}")
eval_labels_dir_sim = os.getenv('EVAL_LABELS_PATH_SIM')
print(f"Eval Sim Labels Path: {eval_labels_dir_sim}")

test_labels_dir_real = os.getenv('TEST_LABELS_PATH_REAL')
print(f"Test Real Labels Path: {test_labels_dir_real}")
test_labels_dir_sim = os.getenv('TEST_LABELS_PATH_SIM')
print(f"Test Sim Labels Path: {test_labels_dir_sim}")

train_files = [os.path.join(train_labels_dir, fname) for fname in os.listdir(train_labels_dir) if fname.endswith('.json')]
eval_files_real = [os.path.join(eval_labels_dir_real, fname) for fname in os.listdir(eval_labels_dir_real) if fname.endswith('.json')]
eval_files_sim = [os.path.join(eval_labels_dir_sim, fname) for fname in os.listdir(eval_labels_dir_sim) if fname.endswith('.json')]
test_files_real = [os.path.join(test_labels_dir_real, fname) for fname in os.listdir(test_labels_dir_real) if fname.endswith('.json')]
test_files_sim = [os.path.join(test_labels_dir_sim, fname) for fname in os.listdir(test_labels_dir_sim) if fname.endswith('.json')]

# Create train and eval datasets
train_dataset = MyMultimodalDataset(data_files=train_files, chat_processor=vl_chat_processor)
train_sampler  = torch.utils.data.RandomSampler(train_dataset,
                                               generator=g,
                                               replacement=False)

eval_dataset_real = MyMultimodalDataset(data_files=eval_files_real, chat_processor=vl_chat_processor)
eval_dataset_sim = MyMultimodalDataset(data_files=eval_files_sim, chat_processor=vl_chat_processor)
test_dataset_real = MyMultimodalDataset(data_files=test_files_real, chat_processor=vl_chat_processor)
test_dataset_sim = MyMultimodalDataset(data_files=test_files_sim, chat_processor=vl_chat_processor)

# --- Define hyperparameters ---

batch_size = hyperparameters["batch_size"]

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=hyperparameters["num_workers_dataloader"],
    worker_init_fn=worker_init_fn,
    generator=g,
    collate_fn=lambda batch: custom_collate_fn(
        batch,
        pad_keys=["input_ids", "attention_mask", "images_seq_mask"],
        tokenizer=tokenizer),
    pin_memory=True,
    persistent_workers=True,
)


eval_dataloader_real = DataLoader(
    eval_dataset_real,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
)
eval_dataloader_sim = DataLoader(
    eval_dataset_sim,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
)

test_dataloader_real = DataLoader(
    test_dataset_real,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
)
test_dataloader_sim = DataLoader(
    test_dataset_sim,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
)

# --- Add artifacts to wandb ---
dataset_artifact = wandb.Artifact(name="continued_train_data_artifact", type="dataset")
dataset_artifact.add_dir("./train/images")
dataset_artifact.add_dir("./train/labels")
dataset_artifact.add_dir("./eval/real/images")
dataset_artifact.add_dir("./eval/real/labels")
dataset_artifact.add_dir("./eval/simulation/images")
dataset_artifact.add_dir("./eval/simulation/labels")
dataset_artifact.add_dir("./test/real/images")
dataset_artifact.add_dir("./test/real/labels")
dataset_artifact.add_dir("./test/simulation/images")
dataset_artifact.add_dir("./test/simulation/labels")

script_name = os.path.basename(__file__)
code_artifact = wandb.Artifact(name="continued_training_code", type="code")
code_artifact.add_file(script_name)

# --- Define number of epochs ---
max_epochs = hyperparameters["max_epochs"]

# --- Optimizer ---
initial_optimizer_lr = hyperparameters["min_lr"] if hyperparameters["warmup_steps"] > 0 else hyperparameters["learning_rate"]
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=initial_optimizer_lr,
    betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
    weight_decay=hyperparameters["weight_decay"],
    is_paged=True
)

# --- Initialize Learning Rate Scheduler ---
scheduler = None
lr_reduction_monitor = None
scheduler_type = hyperparameters["lr_scheduler_type"]
print(f"Initializing scheduler: {scheduler_type}")

if scheduler_type == "ReduceLROnPlateau":
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=hyperparameters["lr_scheduler_mode"],
        factor=hyperparameters["lr_scheduler_factor"],
        patience=hyperparameters["lr_scheduler_patience"],
        verbose=True,
        min_lr=hyperparameters["min_lr"]
    )
    lr_reduction_monitor = LRReductionMonitor()
elif scheduler_type == "CosineAnnealingWarmRestarts":
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=hyperparameters["cosine_warm_restarts_t_0"],
        T_mult=hyperparameters["cosine_warm_restarts_t_mult"],
        eta_min=hyperparameters["min_lr"],
        verbose=True
    )
elif scheduler_type == "StepLR":
    scheduler = StepLR(
        optimizer,
        step_size=hyperparameters["step_lr_step_size"],
        gamma=hyperparameters["step_lr_gamma"],
        verbose=True
    )
else:
    print(f"No scheduler selected or '{scheduler_type}' is not supported. Training with constant LR.")


wandb.config.update(hyperparameters)
wandb.config.update({
    "optimizer": "AdamW (bnb 8bit paged)",
    "quantization": "4-bit",
})


wandb.watch(model, log="all", log_freq=5)

start_time = time.time()
model.train()

# Initialize global best metric before the training loop
initialize_global_best_metric(eval_mode)

# --- Enable/Disable early stopping ---
early_stopper = EarlyStopping(patience=hyperparameters["early_stopping_patience"]) if hyperparameters["early_stopping_enabled"] else None


# --- Training Loop ---

eval_loss_real = 0.0
eval_loss_sim = 0.0
test_loss_real = 0.0
test_loss_sim = 0.0
global_step = 0
checkpoint_counter = 0

stop_training = False

# Dictionary to store the last known evaluation metrics for archival and logging purposes.
last_known_metrics = {
    "eval_loss_real": 0.0, "overall_eval_acc_real": 0.0,
    "eval_loss_sim": 0.0,  "overall_eval_acc_sim": 0.0,
    "avg_eval_loss": 0.0,  "avg_eval_acc": 0.0,
    "test_loss_real": 0.0, "overall_test_acc_real": 0.0,
    "test_loss_sim": 0.0,  "overall_test_acc_sim": 0.0,
    "avg_test_loss": 0.0,  "avg_test_acc": 0.0
}

for epoch in range(max_epochs): # epochs loop
    num_batches = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_dataloader): # batches loop

        # --- Learning Rate Warmup ---
        if global_step < hyperparameters['warmup_steps']:
            warmup_lr = hyperparameters['min_lr'] + (hyperparameters['learning_rate'] - hyperparameters['min_lr']) * (global_step + 1) / hyperparameters['warmup_steps']
            optimizer.param_groups[0]['lr'] = warmup_lr
        elif global_step == hyperparameters['warmup_steps']:
             optimizer.param_groups[0]['lr'] = hyperparameters['learning_rate']
             print(f"Warmup finished. LR set to {hyperparameters['learning_rate']}")

        outputs, current_batch_size, labels = custom_forward(batch, model)
        logits  = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            
        loss.backward()     

        #log_memory_flex(cluster_flag,"After loss backward:",gpu_id=gpu_id, device_name=device_name)

        # Gradient Clipping
        if hyperparameters["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters["grad_clip_norm"])

        optimizer.step()
        optimizer.zero_grad()

        # --- Step LR Schedulers (that are not ReduceLROnPlateau) ---
        if scheduler_type in ["CosineAnnealingWarmRestarts", "StepLR"] and scheduler is not None:
             if global_step >= hyperparameters['warmup_steps']:
                scheduler.step()

        #log_memory_flex(cluster_flag,"After parameters update:",gpu_id=gpu_id, device_name=device_name)

        total_samples += current_batch_size
        num_batches += 1
        
        # Log current batch loss and LR
        current_actual_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Step {global_step}, LR: {current_actual_lr:.2e}, Loss: {loss.item():.4f}")

        wandb.log({
            "train_loss": loss.item(),
            "learning_rate_actual": current_actual_lr,
            "epoch": epoch + 1,
            "step": global_step,
        })

        del batch, outputs, loss, logits, shift_logits, shift_labels
        torch.cuda.empty_cache()
        
        # Evaluate model
        if global_step > 0 and global_step % hyperparameters["eval_every_n_steps"] == 0:
            
            eval_loss_real, overall_eval_acc_real = evaluate_model(model, eval_dataloader_real, hyperparameters["obstacle_category_weights"])

            # Save manual eval checkpoint
            if (checkpoint_counter < len(hyperparameters["manual_checkpoint_steps"])) and (global_step >= hyperparameters["manual_checkpoint_steps"][checkpoint_counter]):
                checkpoint_counter += 1
                manual_eval_chkpoint = os.path.join(hyperparameters["checkpoint_dir"], f"manual_eval_chkpoint_{global_step}/")
                model.save_pretrained(manual_eval_chkpoint)
                print(f"Manual eval checkpoint saved at step {global_step}")
                metadata = {
                    "eval_mode": eval_mode,
                    "step": global_step,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(os.path.join(manual_eval_chkpoint, "checkpoint_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                wandb.log({"Manual_eval_checkpoint_saved": True})
            
            print(f"Evaluation Real - Step {global_step}, Eval Loss Real: {eval_loss_real:.4f}, Eval Overall Accuracy Real: {overall_eval_acc_real:.4f}")

            eval_loss_sim, overall_eval_acc_sim = evaluate_model(model, eval_dataloader_sim, hyperparameters["obstacle_category_weights"])

            print(f"Evaluation Simulation - Step {global_step}, Eval Loss Sim: {eval_loss_sim:.4f},  Eval Overall Accuracy Sim: {overall_eval_acc_sim:.4f}")
            
            avg_eval_loss = (eval_loss_real*hyperparameters["real_image_eval_weight"] + eval_loss_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])
            avg_eval_acc = (overall_eval_acc_real*hyperparameters["real_image_eval_weight"] + overall_eval_acc_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            test_loss_real, overall_test_acc_real = evaluate_model(model, test_dataloader_real, hyperparameters["obstacle_category_weights"])
            print(f"Test Real - Step {global_step}, Test Loss Real: {test_loss_real:.4f}, Test Overall Accuracy Real: {overall_test_acc_real:.4f}")

            test_loss_sim, overall_test_acc_sim = evaluate_model(model, test_dataloader_sim, hyperparameters["obstacle_category_weights"])
            print(f"Test Sim - Step {global_step}, Test Loss Sim: {test_loss_sim:.4f}, Test Overall Accuracy Sim: {overall_test_acc_sim:.4f}")

            avg_test_loss = (test_loss_real*hyperparameters["real_image_eval_weight"] + test_loss_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])
            avg_test_acc = (overall_test_acc_real*hyperparameters["real_image_eval_weight"] + overall_test_acc_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            wandb.log({
                "eval_loss_real": eval_loss_real, "eval_loss_sim": eval_loss_sim, "avg_eval_loss": avg_eval_loss,
                "test_loss_real": test_loss_real, "test_loss_sim": test_loss_sim, "avg_test_loss": avg_test_loss,
                "overall_eval_acc_real": overall_eval_acc_real, "overall_eval_acc_sim": overall_eval_acc_sim,
                "overall_test_acc_real": overall_test_acc_real, "overall_test_acc_sim": overall_test_acc_sim,
                "avg_test_acc": avg_test_acc,
                "step": global_step
            })

            if eval_mode == "loss":
                evaluation_metric = avg_eval_loss
                print(f"Using Average Eval Loss ({avg_eval_loss:.4f}) for LR scheduling and early stopping.")
            else: # accuracy
                evaluation_metric = avg_eval_acc
                print(f"Using Average Eval Accuracy ({avg_eval_acc:.4f}) for LR scheduling and early stopping.")
            
            wandb.log({"main_evaluation_metric": evaluation_metric})
            
            metric_improved_and_checkpoint_saved = save_checkpoint_if_improved(
                    current_metric_value=evaluation_metric,
                    model=model,
                    step=global_step,
                    epoch=epoch + 1,
                    checkpoint_path=best_checkpoint_path,
                    mode=eval_mode,
                    delta=hyperparameters["improvement_delta"]
                ) 
            
            if scheduler_type == "ReduceLROnPlateau" and scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr'] 
                scheduler.step(evaluation_metric) 
                new_lr = optimizer.param_groups[0]['lr'] 
                if lr_reduction_monitor:
                    lr_reduction_monitor.compare_lr(old_lr, new_lr, global_step, epoch)
            
            if early_stopper:
                early_stopper.check(metric_improved_and_checkpoint_saved) 
                if early_stopper.patience_met:
                    if scheduler_type == "ReduceLROnPlateau" and lr_reduction_monitor:
                        if lr_reduction_monitor.lr_reduction_count >= hyperparameters["max_lr_reductions"]:
                            print(f"Early stopping triggered: LR reduced {lr_reduction_monitor.lr_reduction_count}/{hyperparameters['max_lr_reductions']} times and {eval_mode} metric has not improved for {early_stopper.patience} evaluations.")
                            stop_training = True
                        else:
                            print(f"{eval_mode.capitalize()} metric hasn't improved, but max LR reductions not reached ({lr_reduction_monitor.lr_reduction_count}/{hyperparameters['max_lr_reductions']}). Continuing.")
                    else: 
                        print(f"Early stopping triggered: {eval_mode.capitalize()} metric has not improved for {early_stopper.patience} evaluations.")
                        stop_training = True
            
            last_known_metrics.update({
                "eval_loss_real": eval_loss_real, "overall_eval_acc_real": overall_eval_acc_real,
                "eval_loss_sim": eval_loss_sim, "overall_eval_acc_sim": overall_eval_acc_sim,
                "avg_eval_loss": avg_eval_loss, "avg_eval_acc": avg_eval_acc,
                "test_loss_real": test_loss_real, "overall_test_acc_real": overall_test_acc_real,
                "test_loss_sim": test_loss_sim, "overall_test_acc_sim": overall_test_acc_sim,
                "avg_test_loss": avg_test_loss, "avg_test_acc": avg_test_acc
            })

        global_step += 1

        if stop_training:
            break

    if stop_training:
        print(f"Step {global_step} at Epoch {epoch+1}: Early stopping triggered. Breaking epoch loop.")
        break

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{max_epochs} completed - Time: {epoch_time:.2f} seconds")
   
    wandb.log({
        "epoch_time": epoch_time,
        "epoch": epoch + 1
    })
    
    # Save first epoch checkpoint
    if epoch == 0:
        first_epoch_chkpoint = os.path.join(hyperparameters["checkpoint_dir"], "first_epoch_chkpoint/")
        model.save_pretrained(first_epoch_chkpoint)
        print(f"First epoch checkpoint saved.")
        metadata = {
            "eval_mode": eval_mode,
            "epoch": epoch+1,
            "step": global_step,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(first_epoch_chkpoint, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        wandb.log({"first_epoch_checkpoint_saved": True})

# --- Log artifacts to wandb ---
run.log_artifact(dataset_artifact)
run.log_artifact(code_artifact)

# Final save logic
if stop_training:
    print(f"Training stopped early at step {global_step}. Best model is in {best_checkpoint_path}.")
    wandb.log({
        "training_outcome": "early_stopped",
        "final_step_at_stop": global_step,
        "best_model_location": best_checkpoint_path
    })
else:
    print(f"Training completed all {max_epochs} epochs.")
    if hyperparameters["early_stopping_enabled"]:
        print(f"Best model saved at {best_checkpoint_path}")
        wandb.log({"training_outcome": "completed_with_early_stopping_on"})
    else:
        # Save final model if not using early stopping
        final_checkpoint_path = os.path.join(hyperparameters["checkpoint_dir"], "final_save/")
        model.save_pretrained(final_checkpoint_path)
        print(f"Final model saved at {final_checkpoint_path}")
        metadata = {
            "eval_mode": eval_mode, "epoch": epoch+1, "step": global_step,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        metadata.update({f"final_{k}": v for k, v in last_known_metrics.items()})
        with open(os.path.join(final_checkpoint_path, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        wandb.log({"training_outcome": "completed_full_run"})

wandb.finish() 