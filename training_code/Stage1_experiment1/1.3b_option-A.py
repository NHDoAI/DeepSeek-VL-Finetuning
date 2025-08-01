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
from peft import get_peft_model, LoraConfig
from deepseek_vl.utils.io import load_pil_images
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR

import numpy as np

import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cluster_flag = True # Flag to control between cluster and local training
model_type = "1.3B"
eval_mode = "loss" # "loss" or "accuracy"; accuracy is nice to have, but when tested was not as "good" as loss - leads to overfitting


# ------ Configuration & Hyperparameters ------

# trained in three phases, each phase has its own set of hyperparameters
phase_2_start = 500 #start of phase without including warmup steps. For example: if this variable is 500 and each warm up phase is 100 steps, then phase 2 starts at (warm-up + 500) = step 600
phase_3_start = 1000 # phase 3 starts at step 1200

hyperparameters = {

    "base_checkpoint_dir": "./", # Base directory for checkpoints saving
    "run_name_prefix": "1.3b/Option-A_increased_weighs_s42", # Prefix for run names and checkpoint subdirs
    "eval_every_n_steps": 500, # Will be overridden by phase-specific settings
    "master_seed": 42, # seed for random number generation
    "num_workers_dataloader": 4, # number of workers for dataloader
    "lora_rank": 6, # rank of LoRA adapters
    "lora_alpha": 12, # alpha of LoRA adapters
    "lora_dropout": 0.05, # Dropout rate for LoRA adapters
    "batch_size": 6, # batch size
    "max_epochs": 30, #setting maximum number of epochs run
    "min_lr": 1e-6, # minimum learning rate
    "max_lr_reductions": 4, # maximum number of learning rate reduction steps before early stopping is considered. This is one condition for early stopping.
    "early_stopping_enabled_globally": True, # Master switch for dynamic early stopping
    "early_stopping_active_ranges": [[2500, 20000]], # List of [start_step, end_step] ranges in which early stopping is active. Example: [[0, 2000], [4000, 6000]]

    "wandb_project_name": "deepseek-vl-training_final_version", # project name underwhich the run will be saved on wandb
    "real_image_eval_weight": 1.0, # weight for real image evaluation loss
    "sim_image_eval_weight": 0.75, # weight for simulated image evaluation loss

    # Mode-dependent parameters using inline conditionals
    "lr_scheduler_mode": "max" if eval_mode == "accuracy" else "min", # automatically selects mode for learning rate scheduler based on if loss or accuracy is used for evaluation
    "lr_scheduler_patience": 2 if eval_mode == "accuracy" else 1, # setting patience for learning rate scheduler based on eval mode
    "lr_scheduler_factor": 0.5, # how much of old learning rate to keep when reducing (0.5 = 50% of old LR)
    "early_stopping_patience": 8 if eval_mode == "accuracy" else 1, # how many eval steps in which there is no improvement before early stopping is considered. This is the second condition for early stopping.
    "manual_checkpoint_steps":[2199, 2299], # steps at which to manually save checkpoints
    "manual_eval_save": True, # flag to enable manual saving at certain steps, defined by "manual_checkpoint_steps"

    "improvement_delta": 0.001 if eval_mode == "accuracy" else 0.0001,
    "weight_decay": 0.01,  # Initialized to Phase 1, will be managed by phase logic
    "beta1": 0.9,          # Added: Beta1 for AdamW
    "beta2": 0.95,        # Added: Beta2 for AdamW
    "grad_clip_norm": 1.0, # Initialized to Phase 1, will be managed by phase logic

    # Phase-specific hyperparameters
    "learning_rate_phases": [2e-4, 5e-5, 5e-5], # LR for Main Phase 1, Main Phase 2, Main Phase 3
    "weight_decay_phases": [0.0, 0.0, 0.0], # WD for Main Phase 1, Main Phase 2, Main Phase 3 (and their warmups)
    "grad_clip_norm_phases": [1.0, 1.0, 1.0],   # Grad Clip for Main Phase 1, Main Phase 2, Main Phase 3 (and their warmups)
    "phase_boundaries": [phase_2_start, phase_3_start], # conceptual_global_step counts where main phases *would* change if no warmups

    # --- Warmup Phase Hyperparameters ---
    "warmup_steps_phases": [100, 100, 100], # Number of warmup steps for  the phases: WU1, WU2, WU3 respectively

    # --- Phase-specific Learning Rate Scheduler Hyperparameters (for Main Phases) ---
    "lr_scheduler_types_phases": ["CosineAnnealingWarmRestarts", "StepLR", "CosineAnnealingWarmRestarts"], # Scheduler type for each main phase, available: ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR

    # Parameters for CosineAnnealingWarmRestarts (used if "CosineAnnealingWarmRestarts" is selected for a main phase)
    "cosine_warm_restarts_t_0_phases": [100, 100, 100], # Example T_0 values for each main phase
    "cosine_warm_restarts_t_mult_phases": [1, 1, 1],      # Example T_mult values for each main phase

    # Parameters for StepLR (used if "StepLR" is selected for a main phase)
    "step_lr_step_size_phases": [100, 100, 50], # step_size for StepLR for each main phase
    "step_lr_gamma_phases": [0.75, 0.75, 0.5],       # gamma for StepLR for each main phase

    # --- Phase-specific Token Weights and Eval Frequency (for Main Phases and their Warmups) ---
    "keyword_weight_phases": [2, 3, 3],       # Keyword token weight for each phase (applies to WU and Main); [1.25, 1.5, 1.5] for initial weights
    "background_weight_phases": [1.0, 1.0, 0.5],    # Background token weight for each phase (applies to WU and Main)
    "eval_every_n_steps_phases": [250, 100, 100],   # Evaluation frequency for each main phase (evaluation skipped in WU)


}

# Initialize with Phase 1 values for optimizer and initial clipping
# LR will be min_lr if WU1 has steps, otherwise P1's LR.
hyperparameters["learning_rate"] = hyperparameters["min_lr"] if hyperparameters["warmup_steps_phases"][0] > 0 else hyperparameters["learning_rate_phases"][0]
hyperparameters["weight_decay"] = hyperparameters["weight_decay_phases"][0]
hyperparameters["grad_clip_norm"] = hyperparameters["grad_clip_norm_phases"][0]
# Initialize phase-dependent token weights and eval frequency (for WU1/P1)
hyperparameters["keyword_weight"] = hyperparameters["keyword_weight_phases"][0]
hyperparameters["background_weight"] = hyperparameters["background_weight_phases"][0]
hyperparameters["eval_every_n_steps"] = hyperparameters["eval_every_n_steps_phases"][0]


# Construct wandb_run_name, checkpoint_dir and best_checkpoint_path using other hyperparameters
hyperparameters["wandb_run_name"] = f"{hyperparameters['run_name_prefix']}_{eval_mode}"
hyperparameters["checkpoint_dir"] = os.path.join(
    hyperparameters["base_checkpoint_dir"],
    f"{hyperparameters['run_name_prefix']}_{eval_mode}/"
)


# ------ Define useful functions and classes ------

# --- Functions to log memory usage ---
# Useful for monitoring memory usage during training on cluster

# --- Function to get GPU memory info locally when direct access to GPU is available ---
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

# --- Function to log memory usage on cluster when direct access to GPU is not available ---
def log_memory_cluster(prefix=""):
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_memory = properties.total_memory / (1024**2)  # Total memory in bytes
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**2) # Reserved memory in bytes
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2) # Allocated memory in bytes
    print(prefix)
    print(f"{prefix} GPU Memory - Total: {total_memory:.2f} MB | Allocated: {allocated_memory:.2f} MB | Reserved: {reserved_memory:.2f} MB") #| Free: {free_memory:.2f} MB

# --- Function to choose the appropriate memory logging function based on the cluster flag ---
def log_memory_flex(cluster_flag=False, message="GPU Memory:", gpu_id=0, device_name="GPU"):
    if not cluster_flag:
        print(f"CUDA Device Name: {device_name}")
        gpu_memory = get_gpu_memory_info(gpu_id)
        print(f"PyTorch is using GPU {gpu_id}:")
        print(gpu_memory)
    else:
        log_memory_cluster(message)



# --- Function to split data files into training and evaluation sets if data is all packed together, currently not used since data was manually split---
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

        pil_images = load_pil_images(conversation)

        # chat_processor is a VLChatProcessor instance that handles image and text processing from the deepseek_vl.models module
        inputs = self.chat_processor( 
            conversations=conversation,
            images=pil_images,
            force_batchify=True  # Ensures outputs are in a batch-ready format.
        )
        return inputs
    
# --- Custom collate function for padding and concatenating tensors ---
def custom_collate_fn(batch, pad_keys=None, pad_values=None, tokenizer=None):
    """
    Collate a list of samples (each a dict) into a batch.
    
    Args:
        batch (list): List of sample dictionaries.
        pad_keys (list, optional): List of keys that need to be padded along dimension 1.
                                   Default: ["input_ids", "attention_mask", "images_seq_mask"]
        pad_values (dict, optional): A dictionary mapping each pad key to a pad value.
                                     Default for:
                                        "input_ids": batch[0]["pad_id"] (if available, otherwise 0),
                                        "attention_mask": 0,
                                        "images_seq_mask": 0.
    
    Returns:
        dict: A batch dictionary with padded tensors for keys in pad_keys and concatenated tensors for others.
    """
    # Set defaults if not provided.
    if pad_keys is None:
        pad_keys = ["input_ids", "attention_mask", "images_seq_mask"]
    if pad_values is None:
        pad_values = {}
        for key in pad_keys:
            if key == "input_ids":
                pad_values[key] = tokenizer.eos_token_id if tokenizer is not None else 0
            elif key in ["attention_mask", "images_seq_mask"]:
                pad_values[key] = 0
            else:
                pad_values[key] = 0

    padded_batch = {}
    for key in batch[0].keys():
        # If this key should be padded and is a tensor.
        if key in pad_keys and isinstance(batch[0][key], torch.Tensor):
            # Compute maximum sequence length along dimension 1 (the sequence dimension).
            max_len = max(sample[key].shape[1] for sample in batch)
            padded_tensors = []
            for sample in batch:
                tensor = sample[key]
                seq_len = tensor.shape[1]
                pad_size = max_len - seq_len
                if pad_size > 0:
                    pad = torch.full((tensor.shape[0], pad_size), pad_values[key], dtype=tensor.dtype)
                    tensor = torch.cat([tensor, pad], dim=1)
                padded_tensors.append(tensor)
            # Concatenate padded tensors along the batch dimension (dimension 0).
            padded_batch[key] = torch.cat(padded_tensors, dim=0)
        elif isinstance(batch[0][key], torch.Tensor):
            # For tensor keys that do NOT require padding (e.g., pixel_values, images_emb_mask)
            padded_batch[key] = torch.cat([sample[key] for sample in batch], dim=0)
        else:
            # For non-tensor keys, just put them in a list.
            padded_batch[key] = [sample[key] for sample in batch]
    return padded_batch

# --- custom forward pass function for the model ---
def custom_forward(batch, model):
    current_batch_size = batch["input_ids"].shape[0] # current_batch_size is the number of samples in the batch, stored on CPU
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} # move all tensors in the batch to the same device as the model
    if 'pixel_values' in batch:
        batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
    inputs_embeds = model.prepare_inputs_embeds(**batch) # method to prepare inputs embeddings from the batch's input_ids and pixel_values

    del batch["pixel_values"] # remove pixel_values (since they are already processed) to save on GPU memory
    labels = batch["input_ids"].clone() # clone the input_ids to create the labels tensor
    # --- Dynamic masking based on "Assistant:" ---
    # Tokenizer should be in the global scope or passed to this function
    # Assuming `tokenizer` is the global VLChatProcessor's tokenizer
    assistant_phrase = "Assistant:"
    # Encode the assistant_phrase to get token IDs of "Assistant:", without adding special BOS/EOS tokens for the phrase itself
    assistant_token_ids = tokenizer.encode(assistant_phrase, add_special_tokens=False)
    assistant_token_ids_tensor = torch.tensor(assistant_token_ids, device=model.device, dtype=torch.long)
    
    L_assistant = len(assistant_token_ids) # length of the assistant_token_ids tensor
    current_input_ids = batch["input_ids"] # Create a list to store input_ids of each sample in the batch

    for i in range(current_input_ids.shape[0]): # Iterate over batch samples
        sample_input_ids = current_input_ids[i] # Current sample's input_ids
        found_match_for_sample = False
        # Search for the "Assistant:" token sequence in the current sample
        for k_start_index in range(sample_input_ids.size(0) - L_assistant + 1): # from first token position to the end of the input_ids minus length of search phrase plus 1
            window = sample_input_ids[k_start_index : k_start_index + L_assistant] # cut out a window of input_ids of same size as search phrase 
            if torch.equal(window, assistant_token_ids_tensor): # check if window and search phrase is equal
                # Found "Assistant:". Mask labels up to the end of this phrase.
                # The first token of the actual model output starts *after* "Assistant:".
                # So, indices from 0 to (k_start_index + L_assistant - 1) are masked.
                mask_end_exclusive_index = k_start_index + L_assistant
                labels[i, :mask_end_exclusive_index] = -100    # labels in batch i, input ids from beginning to end of search phrase             
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
    labels = labels.masked_fill(batch["attention_mask"] == 0, -100) # assign -100 (ignore_index) to the labels where the attention_mask is 0 (padding tokens)

    # Forward pass through language model
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch['attention_mask'],
        use_cache=False,
        #labels= batch["labels"]  
    )
    return outputs, current_batch_size, labels

def weighted_loss_calculation(logits, labels, sentinel_ids, keyword_weight, background_weight):

    weights = torch.full_like(labels, background_weight, dtype=torch.float32)  # create weight mask of same shape and data type as labels but has values of background_weight
    # Iterate through each type of sentinel (e.g., "<LANE>", "<OBS>")
    for token, sent_id in sentinel_ids.items():                              
        idxs = (labels == sent_id).nonzero(as_tuple=False)                 
        for batch_idx, sentinel_idx in idxs:                                              
            # Apply keyword_weight to the sentinel and the token immediately following the sentinel
            if sentinel_idx + 1 < labels.size(1):                                 
                weights[batch_idx, sentinel_idx : sentinel_idx + 1] = keyword_weight                     

    # shift for causal LM                                              
    shift_logits  = logits[:, :-1, :].contiguous()                     
    shift_labels  = labels[:, 1:].contiguous()                         
    shift_weights = weights[:, 1:].contiguous() # Shift weights accordingly             
    valid_mask    = (shift_labels != -100)    # Mask for non-padding tokens in labels  
    # Calculate loss per token                       
    loss_per_tok  = F.cross_entropy(                                   
        shift_logits.view(-1, shift_logits.size(-1)),                  
        shift_labels.view(-1),                                         
        reduction='none',                                              
        ignore_index=-100                                              
    ).view_as(shift_labels)                                            
    # Apply weights and valid_mask
    # Only consider tokens that are not padding AND are part of the valid sequence
    numerator   = (loss_per_tok * shift_weights * valid_mask).sum()    
    denominator = (shift_weights * valid_mask).sum() + 1e-10
    loss = numerator / denominator  

    return loss

# --- Function to evaluate the model's performance ---

def compute_sentinel_accuracy(logits: torch.Tensor, labels: torch.Tensor, sentinel_ids: dict):
    """
    Returns two dicts keyed by sentinel: num_correct, num_total
    """
    with torch.no_grad():
        # predictions for position t+1 live in logits[:, t, :]
        shift_logits = logits[:, :-1, :]            # [B, T-1, V]
        shift_labels = labels[:, 1:]                # ground-truth for t+1
        preds        = shift_logits.argmax(-1)      # [B, T-1]

        correct_count = {tok: 0 for tok in sentinel_ids}
        total_count   = {tok: 0 for tok in sentinel_ids}

        for tok, tok_id in sentinel_ids.items():
            # positions where *current* token is the sentinel
            mask = labels[:, :-1] == tok_id         # [B, T-1] boolean
            if mask.any():
                total_count[tok]   += mask.sum().item()
                correct_count[tok] += (preds[mask] == shift_labels[mask]).sum().item()

        return correct_count, total_count
    
def evaluate_model(model, data_dataloader):
    #add device argument if needed
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The model to evaluate.
        eval_dataloader: DataLoader for the evaluation dataset.
        device: The device to run evaluation on.
        
    Returns:
        float: Average evaluation loss.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    global_correct = {tok: 0 for tok in sentinel_ids}
    global_total   = {tok: 0 for tok in sentinel_ids}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_dataloader):
            outputs, current_batch_size, labels = custom_forward(batch, model)
            logits  = outputs.logits 
            loss = weighted_loss_calculation(logits, labels, sentinel_ids, hyperparameters["keyword_weight"], hyperparameters["background_weight"])

            total_loss += loss.item()*current_batch_size # accumulate the loss for all batches, to deal with batches that have different sizes (for example the last batch might not have enough samples to fill the batch). loss.item() is the average loss for the current batch
            total_samples += current_batch_size

            correct_count, total_count = compute_sentinel_accuracy(logits, labels, sentinel_ids)

            for sent_tok in sentinel_ids:
                global_correct[sent_tok] += correct_count[sent_tok]
                global_total[sent_tok] += total_count[sent_tok]

        # deleting variables on GPU to free up memory, this ensure when the next batch is loaded, the GPU memory is not full from objects still stored by pytorch in cache
            del batch
            del outputs
            del loss
            torch.cuda.empty_cache() # empty the cache to free up memory

    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    sent_acc = {sent_tok: (global_correct[sent_tok] / global_total[sent_tok] if global_total[sent_tok] else 0.0)
                for sent_tok in sentinel_ids}
    overall_acc = (sum(global_correct.values()) /
                 (sum(global_total.values()) + 1e-9))

    model.train() # set the model back to training mode
    return avg_loss, sent_acc, overall_acc



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
    def __init__(self, patience=3): 
        self.patience = patience # patience is set to "early_stopping_patience" from the hyperparameters
        self.counter = 0
        self.patience_met = False


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

class LRReductionMonitor: # class to monitor how many times the learning rate is reduced
    def __init__(self):
        self.lr_reduction_count = 0
    def compare_lr(self, old_lr, new_lr):
        if new_lr < old_lr: # Compare with LR before stepping
            print(f"Learning rate reduced from {old_lr} to {new_lr} at step {global_step} by ReduceLROnPlateau based on {eval_mode} metric: {evaluation_metric:.4f}")
            self.lr_reduction_count += 1
            wandb.log({"lr_reduction_count": self.lr_reduction_count, "step": global_step, "epoch": epoch + 1})
        elif new_lr == old_lr:
                print("Current scheduler is ReduceLROnPlateau and LR didn't change.")
    

# 1) pick ONE master seed for the whole run
MASTER_SEED = hyperparameters["master_seed"] 

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

# ------ Calculate actual segment start steps ------

# Current number of evaluation-step.

current_eval_count = 0

# These define the global_step at which each segment (Warmup or Main Phase) begins.
segment_start_steps = {}
conceptual_pb = hyperparameters["phase_boundaries"]
warmup_steps = hyperparameters["warmup_steps_phases"]

segment_start_steps["WU1"] = 0
segment_start_steps["P1"] = segment_start_steps["WU1"] + warmup_steps[0]

segment_start_steps["WU2"] = conceptual_pb[0] + warmup_steps[0]
segment_start_steps["P2"] = segment_start_steps["WU2"] + warmup_steps[1]

segment_start_steps["WU3"] = conceptual_pb[1] + warmup_steps[0] + warmup_steps[1]
segment_start_steps["P3"] = segment_start_steps["WU3"] + warmup_steps[2]

# For convenience, create a sorted list of segment names and their start steps
# This helps in determining the current segment in the training loop.
sorted_segment_transitions = sorted(segment_start_steps.items(), key=lambda item: item[1])
# Example: [('WU1', 0), ('P1', 100), ('WU2', 4100), ('P2', 4250), ('WU3', 5250), ('P3', 5300)]

print("Calculated Segment Start Steps:")
for name, step_val in sorted_segment_transitions:
    print(f"  {name}: {step_val}")

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

# --- 8 bit version ---

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # Alternatively, use load_in_8bit=True if preferred
#     llm_int8_skip_modules=["vision_model", "aligner"]  # Skip quantizing vision and aligner modules
# )

# --- 4 bit version ---

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Note: llm_int8_skip_modules is ignored in 4-bit mode.
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

new_tokens = ["<LANE>", "<OBS>", "<DEC>"]  

tokenizer.add_special_tokens(                                       
    {"additional_special_tokens": new_tokens}                       
)       

sentinel_ids = {tok: tokenizer.convert_tokens_to_ids(tok)           
                for tok in new_tokens}                              
keyword_weight     = hyperparameters["keyword_weight"]                                            
background_weight  = hyperparameters["background_weight"]                                            
# you must resize *before* loading LoRA (adds new rows to the LM)    
model.language_model.resize_token_embeddings(len(tokenizer))    

# ---- MONKEY PATCH START ----
# This is necessary because the MultiModalityCausalLM class does not
# implement get_input_embeddings and get_output_embeddings, which PEFT
# needs when saving the model, especially if embeddings are in modules_to_save.
# We forcefully overwrite these methods on the instance.
# This version attempts to directly access the embedding layers from the language_model.
import types

def get_input_embeddings_direct_access(self_mm_model):
    """
    Directly accesses the input embeddings from the language_model,
    assuming it's a standard HF model structure (e.g., LlamaForCausalLM).
    """
    if hasattr(self_mm_model, 'language_model'):
        lang_model = self_mm_model.language_model
        if hasattr(lang_model, 'model') and hasattr(lang_model.model, 'embed_tokens'):
            # Common path for models like LlamaForCausalLM -> language_model.model.embed_tokens
            return lang_model.model.embed_tokens
        elif hasattr(lang_model, 'embed_tokens'):
            # Common path for models like LlamaModel (if language_model was just the base model)
            return lang_model.embed_tokens
        # Add other known paths if necessary for different architectures, e.g., GPT-2 style:
        # elif hasattr(lang_model, 'transformer') and hasattr(lang_model.transformer, 'wte'):
        #     return lang_model.transformer.wte
    raise AttributeError(
        "MultiModalityCausalLM's language_model does not have the expected attribute structure "
        "for input embeddings (e.g., language_model.model.embed_tokens or language_model.embed_tokens)."
    )

model.get_input_embeddings = types.MethodType(get_input_embeddings_direct_access, model)
print("Monkey-patched get_input_embeddings onto the MultiModalityCausalLM instance using direct attribute access.")

def get_output_embeddings_direct_access(self_mm_model):
    """
    Directly accesses the output embeddings (lm_head) from the language_model,
    assuming it's a standard HF CausalLM model.
    """
    if hasattr(self_mm_model, 'language_model') and hasattr(self_mm_model.language_model, 'lm_head'):
        # Common path for CausalLM models -> language_model.lm_head
        return self_mm_model.language_model.lm_head
    # Fallback for models where get_output_embeddings might not be explicitly defined
    # or lm_head is not the direct attribute, but its own method might work (less likely given the issue)
    elif hasattr(self_mm_model, 'language_model') and hasattr(self_mm_model.language_model, 'get_output_embeddings'):
        try:
            # Try calling the language_model's own method as a last resort
            output_embeds = self_mm_model.language_model.get_output_embeddings()
            if output_embeds is not None:
                print("Warning: Used language_model.get_output_embeddings() as fallback for monkey-patch.")
                return output_embeds
        except NotImplementedError:
            pass # If it raises, we fall through to the AttributeError
    raise AttributeError(
        "MultiModalityCausalLM's language_model does not have the expected lm_head attribute "
        "or a working get_output_embeddings method for output embeddings."
    )

model.get_output_embeddings = types.MethodType(get_output_embeddings_direct_access, model)
print("Monkey-patched get_output_embeddings onto the MultiModalityCausalLM instance using direct attribute access.")
# ---- MONKEY PATCH END ----

# --- Apply LoRA Adapters to the Modules via configs ---

lora_rank = hyperparameters["lora_rank"]
lora_alpha = hyperparameters["lora_alpha"]
lora_dropout = hyperparameters["lora_dropout"]

if model_type == "1.3B":
    # --- 1.3b model ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head",
                        "gate_proj", "up_proj", "down_proj",
                        "attn.qkv", "attn.proj", "fc1", "fc2","patch_embed.proj", "attn_pool.q", "attn_pool.kv",
                        #"aligner.layers.0", "aligner.layers.2"
                        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["language_model.model.embed_tokens"] # save embeddings layer when new token is introduced.
    )
elif model_type == "7B":
    # --- 7b model ---
    lora_config = LoraConfig( #to be modified
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head", #language model
                        "gate_proj", "up_proj", "down_proj", #language model
                        "attn.qkv", "attn.proj", "fc1", "fc2","patch_embed.proj", "attn_pool.q", "attn_pool.kv", "lin1", "lin2" #vision model
                        "aligner.high_up_proj", "aligner.low_up_proj" #"aligner.layers.1"
                        ], #aligner
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["language_model.model.embed_tokens"] # save embeddings layer when new token is introduced.
    )

# --- Load Model with LoRA Adapters ---
model = get_peft_model(model, lora_config, adapter_name="adapter")

# --- Define checkpoint path for saving the model-checkpoints ---

best_checkpoint_path = os.path.join(hyperparameters["checkpoint_dir"], "best_chkpoint")

# --- Prepare Training Loop ---

# --- Prepare Datasets ---

train_labels_dir = os.getenv('TRAIN_LABELS_PATH') # get the path to the training labels directory from the environment variables
print(train_labels_dir)


eval_labels_dir_real = os.getenv('EVAL_LABELS_PATH_REAL') # get the path to the evaluation labels real samples directory from the environment variables
print(f"Eval Real Labels Path: {eval_labels_dir_real}")
eval_labels_dir_sim = os.getenv('EVAL_LABELS_PATH_SIM') # get the path to the evaluation labels simulation samples directory from the environment variables
print(f"Eval Sim Labels Path: {eval_labels_dir_sim}")


# Split data into train and eval sets, unused since data was split manually
#train_files, eval_files = split_data_files(labels_dir, eval_ratio=0.2)

train_files = [os.path.join(train_labels_dir, fname) for fname in os.listdir(train_labels_dir) if fname.endswith('.json')] # set the path to all individual training labels files (json)

eval_files_real = [os.path.join(eval_labels_dir_real, fname) for fname in os.listdir(eval_labels_dir_real) if fname.endswith('.json')] # set the path to all individual evaluation labels real samples files (json)
eval_files_sim = [os.path.join(eval_labels_dir_sim, fname) for fname in os.listdir(eval_labels_dir_sim) if fname.endswith('.json')] # set the path to all individual evaluation labels simulation samples files (json)

# Create train and eval datasets
train_dataset = MyMultimodalDataset(data_files=train_files, chat_processor=vl_chat_processor)

# create sampler that is deterministic for each seed but still shuffles the data randomly within the seed
train_sampler  = torch.utils.data.RandomSampler(train_dataset,
                                               generator=g,
                                               replacement=False)

# evaluation set does not need to be shuffled
eval_dataset_real = MyMultimodalDataset(data_files=eval_files_real, chat_processor=vl_chat_processor)
eval_dataset_sim = MyMultimodalDataset(data_files=eval_files_sim, chat_processor=vl_chat_processor)


# --- Define hyperparameters ---

batch_size = hyperparameters["batch_size"]

# Before the loop - only create dataloader references but not the actual dataset yet
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=hyperparameters["num_workers_dataloader"],
    worker_init_fn=worker_init_fn,
    generator=g,            # << this is what makes shuffling deterministic
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


# --- Add artifacts to wandb ---
dataset_artifact = wandb.Artifact(name="data_artifact", type="dataset")
dataset_artifact.add_dir("./train/images")
dataset_artifact.add_dir("./train/labels")
dataset_artifact.add_dir("./eval/real/images")
dataset_artifact.add_dir("./eval/real/labels")
dataset_artifact.add_dir("./eval/simulation/images")
dataset_artifact.add_dir("./eval/simulation/labels")
script_name = os.path.basename(__file__)
code_artifact = wandb.Artifact(name="training_code", type="code")
code_artifact.add_file(script_name)

# --- Define number of epochs ---
max_epochs = hyperparameters["max_epochs"]

# --- Scheduler Hyperparameters ---
lr_scheduler_patience = hyperparameters["lr_scheduler_patience"]
lr_scheduler_factor = hyperparameters["lr_scheduler_factor"]
min_lr = hyperparameters["min_lr"]

scheduler_mode = hyperparameters["lr_scheduler_mode"]

# Initial LR for optimizer: min_lr if WU1 has steps, else P1's LR.
initial_optimizer_lr = hyperparameters["min_lr"] if hyperparameters["warmup_steps_phases"][0] > 0 else hyperparameters["learning_rate_phases"][0]
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=initial_optimizer_lr, # Set initial LR
    betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
    weight_decay=hyperparameters["weight_decay_phases"][0], # Initial WD (WU1/P1)
    is_paged=True
) # paged optimizer

# --- Initialize Learning Rate Scheduler ---
scheduler = None # Will be initialized dynamically based on main phase
current_scheduler_type = None # To track the current scheduler type (or "Warmup")

wandb.config.update(
    {
        # optional: hyperparameter logging
        "run_name": hyperparameters["wandb_run_name"],
        "batch_size": hyperparameters["batch_size"],
        "max_epochs": hyperparameters["max_epochs"],
        "learning_rate": hyperparameters["learning_rate"],
        "optimizer": "AdamW (bnb)",
        "quantization": "4-bit",
        "lora_r": hyperparameters["lora_rank"],
        "lora_alpha": hyperparameters["lora_alpha"],
        "lora_dropout": hyperparameters["lora_dropout"],

        "eval_mode_for_stopping_scheduling": eval_mode,
        "lr_scheduler_mode": hyperparameters["lr_scheduler_mode"],
        "lr_scheduler_patience": hyperparameters["lr_scheduler_patience"],
        "lr_scheduler_factor": hyperparameters["lr_scheduler_factor"],
        "early_stopping_patience": hyperparameters["early_stopping_patience"],
        "improvement_delta": hyperparameters["improvement_delta"],

        "min_lr": hyperparameters["min_lr"],
        "max_lr_reductions": hyperparameters["max_lr_reductions"],
        "master_seed": hyperparameters["master_seed"],
        "keyword_weight": hyperparameters["keyword_weight"],
        "background_weight": hyperparameters["background_weight"],
        "initial_weight_decay": hyperparameters["weight_decay"], # Log initial WD
        "beta1": hyperparameters["beta1"],
        "beta2": hyperparameters["beta2"],
        "initial_grad_clip_norm": hyperparameters["grad_clip_norm"], # Log initial GCN
        "learning_rate_phases": hyperparameters["learning_rate_phases"], # Log phase LRs
        "weight_decay_phases": hyperparameters["weight_decay_phases"],   # Log phase WDs
        "grad_clip_norm_phases": hyperparameters["grad_clip_norm_phases"], # Log phase GCNs
        "phase_boundaries": hyperparameters["phase_boundaries"],          # Log conceptual phase boundaries
        "warmup_steps_phases": hyperparameters["warmup_steps_phases"], # Log warmup steps

        # Log new scheduler hyperparameters
        "lr_scheduler_types_phases": hyperparameters["lr_scheduler_types_phases"],
        "cosine_warm_restarts_t_0_phases": hyperparameters["cosine_warm_restarts_t_0_phases"],
        "cosine_warm_restarts_t_mult_phases": hyperparameters["cosine_warm_restarts_t_mult_phases"],
        "step_lr_step_size_phases": hyperparameters["step_lr_step_size_phases"],
        "step_lr_gamma_phases": hyperparameters["step_lr_gamma_phases"],

        # Log new phase-dependent general hyperparameters
        "keyword_weight_phases": hyperparameters["keyword_weight_phases"],
        "background_weight_phases": hyperparameters["background_weight_phases"],
        "eval_every_n_steps_phases": hyperparameters["eval_every_n_steps_phases"],

        # Log new early stopping hyperparameters
        "early_stopping_enabled_globally": hyperparameters.get("early_stopping_enabled_globally", False),
        "early_stopping_active_ranges": hyperparameters.get("early_stopping_active_ranges", []),
        "real_image_eval_weight": hyperparameters["real_image_eval_weight"],
        "sim_image_eval_weight": hyperparameters["sim_image_eval_weight"],
    }
    )

wandb.watch(model, log="all", log_freq=5)

start_time = time.time()
model.train()

# Initialize global best metric before the training loop
initialize_global_best_metric(eval_mode)

# --- Enable/Disable early stopping ---
early_stopper = None # Initialize early_stopper to None. It will be created dynamically.


# --- Training Loop ---

avg_train_loss = 0.0
eval_loss_real = 0.0
eval_loss_sim = 0.0

global_step = 0  # Track total steps across all epochs

applied_segment_details = { # To track the currently applied segment and its properties
    "name": None, # e.g., "WU1", "P1", "WU2", etc.
    "main_phase_idx": -1, # 0, 1, or 2 corresponding to P1, P2, P3
    "is_warmup": False,
    "start_step": -1
}
steps_in_current_phase_for_eval_counter = 0 # Counter for phase-specific evaluation frequency
checkpoint_counter = 0 # Counter for manual checkpoint saving

stop_training = False # initialize stop_training to False

last_eval_loss_real = last_sent_eval_acc_real = last_overall_eval_acc_real = last_eval_loss_sim = last_sent_eval_acc_sim = last_overall_eval_acc_sim = last_avg_eval_loss = last_avg_eval_acc = 0.0 # initialize variables to store last known evaluation metrics, only for archival purposes. Global scope so can be used anywhere (mainly for logging)

# --- Training Loop ---

for epoch in range(max_epochs): # epochs loop

    num_batches = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_dataloader): # batches loop

        # --- Segment Management (Warmup / Main Phase) ---
        current_segment_name = None
        current_main_phase_idx = -1
        current_segment_is_warmup = False
        current_segment_start_step = -1

        # Determine current segment based on global_step
        for i in range(len(sorted_segment_transitions)):
            name, start_step_val = sorted_segment_transitions[i]
            next_start_step_val = float('inf')
            if i + 1 < len(sorted_segment_transitions):
                next_start_step_val = sorted_segment_transitions[i+1][1]
            
            if start_step_val <= global_step < next_start_step_val:
                current_segment_name = name
                current_segment_start_step = start_step_val
                if "WU" in name:
                    current_segment_is_warmup = True
                    current_main_phase_idx = int(name[2:]) - 1 # WU1 -> 0, WU2 -> 1, WU3 -> 2
                else: # P1, P2, P3
                    current_segment_is_warmup = False
                    current_main_phase_idx = int(name[1:]) - 1 # P1 -> 0, P2 -> 1, P3 -> 2
                break
        
        if current_segment_name is None: # Should not happen if logic is correct
            raise RuntimeError(f"Could not determine current segment for global_step {global_step}")

        # Check if segment has changed
        if current_segment_name != applied_segment_details["name"]:
            print(f"\nGlobal Step {global_step}: Transitioning to Segment: {current_segment_name} (Main Phase Index: {current_main_phase_idx}, Warmup: {current_segment_is_warmup})")
            
            # Update common hyperparameters based on the current_main_phase_idx
            # These apply to both warmup and its subsequent main phase
            new_wd = hyperparameters["weight_decay_phases"][current_main_phase_idx]
            new_gcn = hyperparameters["grad_clip_norm_phases"][current_main_phase_idx]
            new_keyword_weight = hyperparameters["keyword_weight_phases"][current_main_phase_idx]
            new_background_weight = hyperparameters["background_weight_phases"][current_main_phase_idx]
            new_eval_every_n_steps = hyperparameters["eval_every_n_steps_phases"][current_main_phase_idx]

            optimizer.param_groups[0]['weight_decay'] = new_wd
            hyperparameters["grad_clip_norm"] = new_gcn
            hyperparameters["keyword_weight"] = new_keyword_weight
            hyperparameters["background_weight"] = new_background_weight
            hyperparameters["eval_every_n_steps"] = new_eval_every_n_steps # Eval only happens in main phase

            log_payload = {
                "current_segment": current_segment_name,
                "current_main_phase_idx": current_main_phase_idx + 1, # 1-indexed for logging
                "is_warmup_segment": 1 if current_segment_is_warmup else 0,
                "segment_weight_decay": new_wd,
                "segment_grad_clip_norm": new_gcn,
                "segment_keyword_weight": new_keyword_weight,
                "segment_background_weight": new_background_weight,
                "segment_eval_every_n_steps": new_eval_every_n_steps,
                "step": global_step,
                "epoch": epoch + 1
            }

            if current_segment_is_warmup:
                scheduler = None # No scheduler during warmup
                current_scheduler_type = "WarmupLinear" # Indicate manual LR control
                # LR is set per step during warmup
                print(f"  Segment Type: Warmup. LR will ramp from {hyperparameters['min_lr']} to {hyperparameters['learning_rate_phases'][current_main_phase_idx]}.")
                print(f"  WD: {new_wd}, Grad Clip: {new_gcn}, Keyword W: {new_keyword_weight}, BG W: {new_background_weight}")
                log_payload["segment_scheduler_type"] = current_scheduler_type
            else: # Main Phase
                new_lr = hyperparameters["learning_rate_phases"][current_main_phase_idx]
                new_scheduler_type_for_main_phase = hyperparameters["lr_scheduler_types_phases"][current_main_phase_idx]
                
                optimizer.param_groups[0]['lr'] = new_lr # Set LR for the main phase
                steps_in_current_phase_for_eval_counter = 0 # Reset eval counter for the new main phase
                
                print(f"  Segment Type: Main Phase. LR: {new_lr}, Scheduler: {new_scheduler_type_for_main_phase}")
                print(f"  WD: {new_wd}, Grad Clip: {new_gcn}, Keyword W: {new_keyword_weight}, BG W: {new_background_weight}, Eval Every: {new_eval_every_n_steps} steps")
                log_payload["segment_learning_rate"] = new_lr
                log_payload["segment_scheduler_type"] = new_scheduler_type_for_main_phase

                # Initialize or Re-initialize Scheduler for the new main phase
                if new_scheduler_type_for_main_phase != current_scheduler_type or scheduler is None:
                    print(f"  Initializing scheduler for Main Phase {current_main_phase_idx + 1}: {new_scheduler_type_for_main_phase}")
                    
                    # Reset initial_lr in optimizer's param groups to ensure the new scheduler
                    # picks up the current LR as its base_lr, not a stale one from a previous phase.
                    for group in optimizer.param_groups:
                        if 'initial_lr' in group:
                            del group['initial_lr']
                    
                    # Initialize scheduler based on the new_scheduler_type_for_main_phase, set according to the hyperparameters
                    if new_scheduler_type_for_main_phase == "ReduceLROnPlateau":
                        scheduler = ReduceLROnPlateau(
                            optimizer,
                            mode=hyperparameters["lr_scheduler_mode"],
                            factor=hyperparameters["lr_scheduler_factor"],
                            patience=hyperparameters["lr_scheduler_patience"],
                            verbose=True,
                            min_lr=hyperparameters["min_lr"]
                        )
                        lr_reduction_monitor = LRReductionMonitor()
                    elif new_scheduler_type_for_main_phase == "CosineAnnealingWarmRestarts":
                        t_0_current_phase = hyperparameters["cosine_warm_restarts_t_0_phases"][current_main_phase_idx]
                        t_mult_current_phase = hyperparameters["cosine_warm_restarts_t_mult_phases"][current_main_phase_idx]
                        scheduler = CosineAnnealingWarmRestarts(
                            optimizer,
                            T_0=t_0_current_phase,
                            T_mult=t_mult_current_phase,
                            eta_min=hyperparameters["min_lr"],
                            verbose=True,
                            last_epoch=-1 # Important for re-initialization
                        )
                    elif new_scheduler_type_for_main_phase == "StepLR":
                        step_size_current_phase = hyperparameters["step_lr_step_size_phases"][current_main_phase_idx]
                        gamma_current_phase = hyperparameters["step_lr_gamma_phases"][current_main_phase_idx]
                        scheduler = StepLR(
                            optimizer,
                            step_size=step_size_current_phase,
                            gamma=gamma_current_phase,
                            verbose=True
                        )
                    else:
                        raise ValueError(f"Unsupported scheduler type: {new_scheduler_type_for_main_phase} for main phase {current_main_phase_idx + 1}")
                    current_scheduler_type = new_scheduler_type_for_main_phase
            
            wandb.log(log_payload)
            
            applied_segment_details = {
                "name": current_segment_name,
                "main_phase_idx": current_main_phase_idx,
                "is_warmup": current_segment_is_warmup,
                "start_step": current_segment_start_step
            }
        # --- End Segment Management ---

        # --- Learning Rate Adjustment for Warmup Segments ---
        if applied_segment_details["is_warmup"]:
            steps_into_warmup = global_step - applied_segment_details["start_step"]
            total_warmup_steps_for_segment = hyperparameters["warmup_steps_phases"][applied_segment_details["main_phase_idx"]]
            target_lr_main_phase = hyperparameters["learning_rate_phases"][applied_segment_details["main_phase_idx"]]
            min_lr_val = hyperparameters["min_lr"]

            if total_warmup_steps_for_segment > 0:
                # Linear ramp-up: current_lr = start_lr + (end_lr - start_lr) * (current_step / total_steps)
                # Add 1 to steps_into_warmup because it's 0-indexed for the first step of warmup
                warmup_lr = min_lr_val + (target_lr_main_phase - min_lr_val) * ( (steps_into_warmup +1) / total_warmup_steps_for_segment)
                optimizer.param_groups[0]['lr'] = warmup_lr
            elif total_warmup_steps_for_segment == 0: # Handle 0 warmup steps (direct jump to target LR)
                 optimizer.param_groups[0]['lr'] = target_lr_main_phase


        outputs, current_batch_size, labels = custom_forward(batch, model)
        logits  = outputs.logits # get the logits from the outputs
        loss = weighted_loss_calculation(logits, labels, sentinel_ids, hyperparameters["keyword_weight"], hyperparameters["background_weight"])
                            
                                 
        loss.backward() # backpropagate the loss after calculation 

        log_memory_flex("After loss backward:",gpu_id=gpu_id, device_name=device_name) # log the gpu memory after the loss backward

        # Gradient Clipping
        if hyperparameters["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters["grad_clip_norm"])

        optimizer.step() # update the model parameters
        optimizer.zero_grad() # reset the gradients

        # --- Step LR Schedulers (only if in Main Phase and scheduler is active) ---
        if not applied_segment_details["is_warmup"] and \
           current_scheduler_type in ["CosineAnnealingWarmRestarts", "StepLR"] and \
           scheduler is not None:
            scheduler.step()

        log_memory_flex("After parameters update:",gpu_id=gpu_id, device_name=device_name)


        total_samples += current_batch_size
        num_batches += 1
        
        steps_in_current_phase_for_eval_counter += 1 # Increments always

        # Log current batch loss and LR
        current_actual_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Step {global_step} (Seg: {applied_segment_details['name']}), LR: {current_actual_lr:.2e}, Loss: {loss.item():.4f}")

        wandb.log({
            "train_loss": loss.item(),
            "learning_rate_actual": current_actual_lr,
            "epoch": epoch + 1,
            "step": global_step,
            "segment_name": applied_segment_details["name"]
        })

        # deleting variables on GPU to free up memory, this ensure when the next batch is loaded, the GPU memory is not full from objects still stored by pytorch in cache
        del batch
        del outputs
        del loss
        torch.cuda.empty_cache() # empty the cache to free up memory
        
        # Evaluate model (only if in a Main Phase and at eval frequency)
        if not applied_segment_details["is_warmup"] and \
           steps_in_current_phase_for_eval_counter % hyperparameters["eval_every_n_steps"] == 0:
            
            #get eval loss and accuracy
            eval_loss_real, sent_eval_acc_real, overall_eval_acc_real = evaluate_model(model, eval_dataloader_real)

            current_eval_count +=1

            # Save manual eval checkpoint
            if (checkpoint_counter < len(hyperparameters["manual_checkpoint_steps"])) and (global_step >= hyperparameters["manual_checkpoint_steps"][checkpoint_counter]): # check if at the current step, a checkpoint should be saved
                checkpoint_counter += 1
                manual_eval_chkpoint = os.path.join(hyperparameters["checkpoint_dir"], f"manual_eval_chkpoint_{current_eval_count}/")
                model.save_pretrained(manual_eval_chkpoint)
                print(f"first epoch checkpoint saved at time:{time.strftime('%Y-%m-%d %H:%M:%S')}")
                # Add metadata about this checkpoint
                metadata = {
                    "eval_mode": eval_mode,
                    "eval count": current_eval_count,
                    "step": global_step,
                    #"train_loss": total_epoch_train_loss / total_samples if total_samples > 0 else 0.0, # Use current epoch's avg train loss
                    "eval_metric_real": last_eval_loss_real if eval_mode == "loss" else last_overall_eval_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
                    "eval_metric_sim": last_eval_loss_sim if eval_mode == "loss" else last_overall_eval_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
                    "avg_eval_metric": last_avg_eval_loss if eval_mode == "loss" else last_avg_eval_acc,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # logging the metadata about the checkpoint
                with open(os.path.join(manual_eval_chkpoint, "checkpoint_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                wandb.log({"Manual_eval_checkpoint_saved": True})
            # print out the evaluation loss and accuracy for the real samples
            print(f"Evaluation Real - Step {global_step}, Eval Loss Real: {eval_loss_real:.4f}, Eval Overall Accuracy Real: {overall_eval_acc_real:.4f}, Eval Key-tokens Accuracy Real: {sent_eval_acc_real}")

            # get the evaluation loss and accuracy for the simulation samples
            eval_loss_sim, sent_eval_acc_sim, overall_eval_acc_sim = evaluate_model(model, eval_dataloader_sim)

            print(f"Evaluation Simulation - Step {global_step}, Eval Loss Sim: {eval_loss_sim:.4f},  Eval Overall Accuracy Sim: {overall_eval_acc_sim:.4f}, Eval Key-tokens Accuracy Sim: {sent_eval_acc_sim}")
            
            #calculate weighted average of eval loss
            avg_eval_loss = (eval_loss_real*hyperparameters["real_image_eval_weight"] + eval_loss_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            #calculate weighted average of eval accuracy
            avg_eval_acc = (overall_eval_acc_real*hyperparameters["real_image_eval_weight"] + overall_eval_acc_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            # Log evaluation metrics
            wandb.log({
                "eval_loss_real": eval_loss_real,
                "eval_loss_sim": eval_loss_sim,
                "avg_eval_loss": avg_eval_loss,
                "sent_eval_acc_real": sent_eval_acc_real,
                "sent_eval_acc_sim": sent_eval_acc_sim,
                "overall_eval_acc_real": overall_eval_acc_real,
                "overall_eval_acc_sim": overall_eval_acc_sim,
                # "current_lr": optimizer.param_groups[0]['lr'], # Already logged per step as learning_rate_actual
                "step": global_step
            })

            # --- Determine if early stopping is active for the current step ---
            is_early_stop_active_now = False
            if hyperparameters.get("early_stopping_enabled_globally", False):
                active_ranges = hyperparameters.get("early_stopping_active_ranges", [])
                # check if the current step is within the active ranges for early stopping, then set is_early_stop_active_now to True
                for start_step_es, end_step_es in active_ranges:
                    if start_step_es <= global_step <= end_step_es:
                        is_early_stop_active_now = True
                        break
            
            if is_early_stop_active_now: # if early stopping is active for the current step
                if early_stopper is None: # if early stopper is not initialized, initialize it
                    early_stopper = EarlyStopping(patience=hyperparameters["early_stopping_patience"])
                    print(f"Global Step {global_step}: Early stopping is NOW ACTIVE. Patience reset.")
                    wandb.log({"early_stopping_status": "activated", "step": global_step, "epoch": epoch + 1})
            else: # if early stopping is not active for the current step
                if early_stopper is not None: # if early stopper is initialized, deactivate it
                    print(f"Global Step {global_step}: Early stopping is NOW INACTIVE.")
                    wandb.log({"early_stopping_status": "deactivated", "step": global_step, "epoch": epoch + 1})
                early_stopper = None 

            # set the type of evaluation metric based on the evaluation mode
            if eval_mode == "loss": # if the evaluation mode is loss, use the average evaluation loss
                evaluation_metric = avg_eval_loss
                print(f"Using Average Eval Loss ({avg_eval_loss:.4f}) for LR scheduling and early stopping.")
            elif eval_mode == "accuracy":
                evaluation_metric = avg_eval_acc
                print(f"Using Average Eval Accuracy ({avg_eval_acc:.4f}) for LR scheduling and early stopping.")
            else:
                raise ValueError(f"Invalid eval_mode: {eval_mode}. Please choose 'loss' or 'accuracy'.")
            
            # save the checkpoint if the current best evaluation metric has improved, also return true if the metric has improved
            metric_improved_and_checkpoint_saved = save_checkpoint_if_improved(
                    current_metric_value=evaluation_metric,
                    model=model,
                    step=global_step,
                    epoch=epoch + 1,
                    checkpoint_path=best_checkpoint_path,
                    mode=eval_mode,
                    delta=hyperparameters["improvement_delta"]
                ) 
            
            if current_scheduler_type == "ReduceLROnPlateau" and scheduler is not None: # Already guarded by not applied_segment_details["is_warmup"]
                old_lr = optimizer.param_groups[0]['lr'] 
                scheduler.step(evaluation_metric) 
                new_lr = optimizer.param_groups[0]['lr'] 
                lr_reduction_monitor.compare_lr(old_lr, new_lr)
            
            if is_early_stop_active_now and early_stopper is not None:
                early_stopper.check(metric_improved_and_checkpoint_saved) # updating early stopper parameters based on result of metric_improved_and_checkpoint_saved

                # setting up early stopping condition based on early stopping patience and learning rate reduction
                if early_stopper.patience_met:
                    if current_scheduler_type == "ReduceLROnPlateau" and scheduler is not None:
                        if lr_reduction_monitor.lr_reduction_count >= hyperparameters["max_lr_reductions"]:
                            print(f"Early stopping triggered: LR reduced {lr_reduction_monitor.lr_reduction_count}/{hyperparameters['max_lr_reductions']} times and {eval_mode} metric ({evaluation_metric:.4f}) hasn't improved for {early_stopper.patience} evaluations. Stopping.")
                            stop_training = True
                        else:
                            print(f"{eval_mode.capitalize()} metric ({evaluation_metric:.4f}) hasn't improved for {early_stopper.patience} evaluations, but max LR reductions ({lr_reduction_monitor.lr_reduction_count}/{hyperparameters['max_lr_reductions']}) not reached. Continuing.")
                    else: 
                        current_lr_val = optimizer.param_groups[0]['lr']
                        print(f"{eval_mode.capitalize()} metric ({evaluation_metric:.4f}) hasn't improved for {early_stopper.patience} evaluations. Scheduler is {current_scheduler_type}. Current LR: {current_lr_val}. Early stopping.")
                        stop_training = True
            
            last_eval_loss_real, last_sent_eval_acc_real, last_overall_eval_acc_real, last_eval_loss_sim, last_sent_eval_acc_sim, last_overall_eval_acc_sim, last_avg_eval_loss, last_avg_eval_acc = eval_loss_real, sent_eval_acc_real, overall_eval_acc_real, eval_loss_sim, sent_eval_acc_sim, overall_eval_acc_sim, avg_eval_loss, avg_eval_acc

        global_step += 1 # Increment global_step after all operations for the current step are done

        if stop_training: # break out of the batch loop if early stopping is triggered
            break

    # If early stopping was triggered, break out of the epoch loop too
    if stop_training:
        print(f"Step {global_step} at Epoch {epoch+1}: Early stopping triggered. Breaking epoch loop.")
        break

    epoch_time = time.time() - start_time # calculate the time taken for the epoch
    print(f"Epoch {epoch+1}/{max_epochs} completed - Time: {epoch_time:.2f} seconds") # print the time taken for the epoch
   
    # log the time taken for the epoch
    wandb.log({
        "epoch_time": epoch_time,
        "epoch": epoch + 1
    })
    

    # Save first epoch checkpoint
    if epoch == 0:
        first_epoch_chkpoint = os.path.join(hyperparameters["checkpoint_dir"], "first_epoch_chkpoint/")
        model.save_pretrained(first_epoch_chkpoint)
        print(f"first epoch checkpoint saved at time:{time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Add metadata about this checkpoint
        metadata = {
            "eval_mode": eval_mode,
            "epoch": epoch+1,
            "step": global_step,

            "eval_metric_real": last_eval_loss_real if eval_mode == "loss" else last_overall_eval_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "eval_metric_sim": last_eval_loss_sim if eval_mode == "loss" else last_overall_eval_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "avg_eval_metric": last_avg_eval_loss if eval_mode == "loss" else last_avg_eval_acc,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(first_epoch_chkpoint, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        wandb.log({"first_epoch_checkpoint_saved": True})

# --- Log artifacts to wandb ---
run.log_artifact(dataset_artifact)
run.log_artifact(code_artifact)

# Save the fine-tuned model at the end if early stopping wasn't used
if not hyperparameters.get("early_stopping_enabled_globally", False):
    # This means dynamic early stopping was globally disabled.
    # Save the model as it completed all epochs or was stopped by other means.
    final_checkpoint_path = os.path.join(hyperparameters["checkpoint_dir"], "final_save/")
    model.save_pretrained(final_checkpoint_path)
    
    # Save metadata about the final checkpoint

    metadata = {
            "eval_mode": eval_mode,
            "epoch": epoch+1,
            "step": global_step,

            "eval_metric_real": last_eval_loss_real if eval_mode == "loss" else last_overall_eval_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "eval_metric_sim": last_eval_loss_sim if eval_mode == "loss" else last_overall_eval_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "avg_eval_metric": last_avg_eval_loss if eval_mode == "loss" else last_avg_eval_acc,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    with open(os.path.join(final_checkpoint_path, "checkpoint_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    wandb.log({#modify as needed
            "eval_mode": eval_mode,
            "final_epoch": epoch+1,
            "final_step": global_step,
            #"train_loss": total_epoch_train_loss / total_samples if total_samples > 0 else 0.0, # Use current epoch's avg train loss
            "final_eval_metric_real": last_eval_loss_real if eval_mode == "loss" else last_overall_eval_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "final_eval_metric_sim": last_eval_loss_sim if eval_mode == "loss" else last_overall_eval_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "final_avg_eval_metric": last_avg_eval_loss if eval_mode == "loss" else last_avg_eval_acc,

        })
elif stop_training: # Dynamic early stopping was enabled and it triggered a stop
    print(f"Training was stopped early at step {global_step} due to dynamic early stopping criteria. The 'best' model is already saved in {best_checkpoint_path}.")
    wandb.log({
        "training_outcome": "early_stopped_dynamically",
        "final_epoch_at_stop": epoch + 1,
        "final_step_at_stop": global_step,
        "best_model_location": best_checkpoint_path
    })
else: # Dynamic early stopping was enabled, but training completed all epochs without triggering a stop
    print(f"Training completed all {max_epochs} epochs. Dynamic early stopping was enabled but not triggered. The 'best' model is already saved in {best_checkpoint_path}.")
    wandb.log({
        "training_outcome": "completed_all_epochs_early_stopping_enabled_not_triggered",
        "final_epoch": epoch + 1,
        "final_step": global_step,
        "best_model_location": best_checkpoint_path
    })


wandb.finish()
