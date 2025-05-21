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

cluster_flag = True
model_type = "1.3B"
eval_mode = "accuracy" # "loss" or "accuracy"

#to be modified
#checkpoint_dir = "./1.3b_less-lora_rerun_v7_deterministic_seed/"

# ------ Configuration & Hyperparameters ------

hyperparameters = {

    "base_checkpoint_dir": "./", # Base directory for checkpoints
    "run_name_prefix": "1.3b_test_option-A", # Prefix for run names and checkpoint subdirs
    "eval_every_n_steps": 5,
    "master_seed": 42,
    "num_workers_dataloader": 4,
    "lora_rank": 6,
    "lora_alpha": 12,
    "lora_dropout": 0.05, # Consider changing between 1.3B and 7B models
    "batch_size": 1,
    "max_epochs": 300,
    "min_lr": 1e-6,
    "max_lr_reductions": 3,
    "early_stop_flag": True,
    # "keyword_weight": 1.5,
    # "background_weight": 1.0,
    "wandb_project_name": "deepseek-vl-training_final_version",
    "real_image_eval_weight": 1.0,
    "sim_image_eval_weight": 0.5,

    # Mode-dependent parameters using inline conditionals
    "lr_scheduler_mode": "max" if eval_mode == "accuracy" else "min",
    "lr_scheduler_patience": 2 if eval_mode == "accuracy" else 1,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 5 if eval_mode == "accuracy" else 3,
    "improvement_delta": 0.001 if eval_mode == "accuracy" else 0.0001,
    "weight_decay": 0.01,  # Initialized to Phase 1, will be managed by phase logic
    "beta1": 0.9,          # Added: Beta1 for AdamW
    "beta2": 0.95,        # Added: Beta2 for AdamW
    "grad_clip_norm": 1.0, # Initialized to Phase 1, will be managed by phase logic

    # Phase-specific hyperparameters
    "learning_rate_phases": [2e-4, 3e-5, 2e-5], # LR for Phase 1, Phase 2, Phase 3
    "weight_decay_phases": [0.0, 0.0, 0.0], # WD for Phase 1, Phase 2, Phase 3
    "grad_clip_norm_phases": [1.0, 1.0, 1.0],   # Grad Clip for Phase 1, Phase 2, Phase 3
    "phase_boundaries": [2500, 5000], # global_step counts where phases change

    # --- Phase-specific Learning Rate Scheduler Hyperparameters ---
    "lr_scheduler_types_phases": ["CosineAnnealingWarmRestarts", "StepLR", "StepLR"], # Scheduler type for each phase

    # Parameters for CosineAnnealingWarmRestarts (used if "CosineAnnealingWarmRestarts" is selected for a phase)
    # T_0: Number of iterations for the first restart.
    # T_mult: A factor increases T_i after a restart. T_i = T_i * T_mult.
    # User needs to set these carefully based on phase length and desired restart behavior.
    # Example: If a phase has 5000 steps, T_0=1000, T_mult=1 means 5 restarts. T_0=1000, T_mult=2 means restarts at 1000, 1000+2000=3000.
    "cosine_warm_restarts_t_0_phases": [1000, 1000, 1500], # Example T_0 values for each phase
    "cosine_warm_restarts_t_mult_phases": [1, 1, 1],      # Example T_mult values for each phase

    # Parameters for StepLR (used if "StepLR" is selected for a phase)
    "step_lr_step_size_phases": [1000, 1000, 500], # step_size for StepLR for each phase
    "step_lr_gamma_phases": [0.7, 0.5, 0.5],       # gamma for StepLR for each phase

    # --- Phase-specific Token Weights and Eval Frequency ---
    "keyword_weight_phases": [1.0, 1.5, 1.5],       # Keyword token weight for each phase
    "background_weight_phases": [1.0, 1, 0.5],    # Background token weight for each phase
    "eval_every_n_steps_phases": [500, 250, 100],   # Evaluation frequency for each phase

    # Note: "lr_scheduler_patience", "lr_scheduler_factor", "lr_scheduler_mode", "min_lr",
    # "max_lr_reductions" are still in hyperparameters.
    # "min_lr" is used by ReduceLROnPlateau and CosineAnnealingWarmRestarts.
    # The others are primarily for ReduceLROnPlateau.
}

# Initialize with Phase 1 values for optimizer and initial clipping
hyperparameters["learning_rate"] = hyperparameters["learning_rate_phases"][0]
hyperparameters["weight_decay"] = hyperparameters["weight_decay_phases"][0]
hyperparameters["grad_clip_norm"] = hyperparameters["grad_clip_norm_phases"][0]
# Initialize phase-dependent token weights and eval frequency
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

        pil_images = load_pil_images(conversation)

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

def custom_forward(batch, model):
    current_batch_size = batch["input_ids"].shape[0] # current_batch_size is the number of samples in the batch, stored on CPU
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if 'pixel_values' in batch:
        batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
    inputs_embeds = model.prepare_inputs_embeds(**batch)

    del batch["pixel_values"]
    labels = batch["input_ids"].clone()
    # For each sequence in the batch, mask out the first 1014 tokens
    labels[:, 0:1014] = -100
    labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
    #batch["labels"] = labels

    # Forward pass through language model
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch['attention_mask'],
        use_cache=False,
        #labels= batch["labels"]  
    )
    return outputs, current_batch_size, labels

def weighted_loss_calculation(logits, labels, sentinel_ids, keyword_weight, background_weight):

    weights = torch.full_like(labels, background_weight, dtype=torch.float32)  
    for token, token_id in sentinel_ids.items():                              
        idxs = (labels == token_id).nonzero(as_tuple=False)                 
        for batch_idx, sentinel_idx in idxs:                                              
            if sentinel_idx + 1 < labels.size(1):                                 
                weights[batch_idx, sentinel_idx + 1] = keyword_weight                     

    # shift for causal LM                                              
    shift_logits  = logits[:, :-1, :].contiguous()                     
    shift_labels  = labels[:, 1:].contiguous()                         
    shift_weights = weights[:, 1:].contiguous()                        
    valid_mask    = (shift_labels != -100)                             
    loss_per_tok  = F.cross_entropy(                                   
        shift_logits.view(-1, shift_logits.size(-1)),                  
        shift_labels.view(-1),                                         
        reduction='none',                                              
        ignore_index=-100                                              
    ).view_as(shift_labels)                                            
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

            total_loss += loss.item()*current_batch_size # to deal with batches that have different sizes (for example the last batch might not have enough samples to fill the batch)
            total_samples += current_batch_size

            correct_count, total_count = compute_sentinel_accuracy(logits, labels, sentinel_ids)

            for sent_tok in sentinel_ids:
                global_correct[sent_tok] += correct_count[sent_tok]
                global_total[sent_tok] += total_count[sent_tok]


            del batch
            del outputs
            del loss
            torch.cuda.empty_cache()

    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    sent_acc = {sent_tok: (global_correct[sent_tok] / global_total[sent_tok] if global_total[sent_tok] else 0.0)
                for sent_tok in sentinel_ids}
    overall_acc = (sum(global_correct.values()) /
                 (sum(global_total.values()) + 1e-9))

    model.train()
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
    def compare_lr(self, old_lr, new_lr):
        if new_lr < old_lr: # Compare with LR before stepping
            print(f"Learning rate reduced from {old_lr} to {new_lr} at step {global_step} by ReduceLROnPlateau based on {eval_mode} metric: {evaluation_metric:.4f}")
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
    mode="disabled"
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
        modules_to_save=["language_model.model.embed_tokens"]
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
        modules_to_save=["language_model.model.embed_tokens"]
    )

# --- Load Model with LoRA Adapters ---
model = get_peft_model(model, lora_config, adapter_name="adapter")

# --- Define checkpoint path for saving the model-checkpoints ---

best_checkpoint_path = os.path.join(hyperparameters["checkpoint_dir"], "best_chkpoint")

# --- Prepare Training Loop ---

# --- Prepare Datasets ---

train_labels_dir = os.getenv('TRAIN_LABELS_PATH')
print(train_labels_dir)
# eval_labels_dir = os.getenv('EVAL_LABELS_PATH') # Old
# print(eval_labels_dir)
# test_labels_dir = os.getenv('TEST_LABELS_PATH') # Old
# print(test_labels_dir)

eval_labels_dir_real = os.getenv('EVAL_LABELS_PATH_REAL')
print(f"Eval Real Labels Path: {eval_labels_dir_real}")
eval_labels_dir_sim = os.getenv('EVAL_LABELS_PATH_SIM')
print(f"Eval Sim Labels Path: {eval_labels_dir_sim}")

test_labels_dir_real = os.getenv('TEST_LABELS_PATH_REAL')
print(f"Test Real Labels Path: {test_labels_dir_real}")
test_labels_dir_sim = os.getenv('TEST_LABELS_PATH_SIM')
print(f"Test Sim Labels Path: {test_labels_dir_sim}")


# Split data into train and eval sets
#train_files, eval_files = split_data_files(labels_dir, eval_ratio=0.2)

train_files = [os.path.join(train_labels_dir, fname) for fname in os.listdir(train_labels_dir) if fname.endswith('.json')]
# eval_files = [os.path.join(eval_labels_dir, fname) for fname in os.listdir(eval_labels_dir) if fname.endswith('.json')] # Old
# test_files = [os.path.join(test_labels_dir, fname) for fname in os.listdir(test_labels_dir) if fname.endswith('.json')] # Old

eval_files_real = [os.path.join(eval_labels_dir_real, fname) for fname in os.listdir(eval_labels_dir_real) if fname.endswith('.json')]
eval_files_sim = [os.path.join(eval_labels_dir_sim, fname) for fname in os.listdir(eval_labels_dir_sim) if fname.endswith('.json')]

test_files_real = [os.path.join(test_labels_dir_real, fname) for fname in os.listdir(test_labels_dir_real) if fname.endswith('.json')]
test_files_sim = [os.path.join(test_labels_dir_sim, fname) for fname in os.listdir(test_labels_dir_sim) if fname.endswith('.json')]

# Create train and eval datasets
train_dataset = MyMultimodalDataset(data_files=train_files, chat_processor=vl_chat_processor)
train_sampler  = torch.utils.data.RandomSampler(train_dataset,
                                               generator=g,
                                               replacement=False)

# eval_dataset = MyMultimodalDataset(data_files=eval_files, chat_processor=vl_chat_processor) # Old
# test_dataset = MyMultimodalDataset(data_files=test_files, chat_processor=vl_chat_processor) # Old

eval_dataset_real = MyMultimodalDataset(data_files=eval_files_real, chat_processor=vl_chat_processor)
eval_dataset_sim = MyMultimodalDataset(data_files=eval_files_sim, chat_processor=vl_chat_processor)

test_dataset_real = MyMultimodalDataset(data_files=test_files_real, chat_processor=vl_chat_processor)
test_dataset_sim = MyMultimodalDataset(data_files=test_files_sim, chat_processor=vl_chat_processor)

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

# eval_dataloader = DataLoader( # Old
#     eval_dataset, 
#     batch_size=batch_size,
#     shuffle=False, 
#     collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
# )
# test_dataloader = DataLoader( # Old
#     test_dataset, 
#     batch_size=batch_size, 
#     shuffle=False, 
#     collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
# )

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
dataset_artifact = wandb.Artifact(name="test_data_artifact", type="dataset")
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
code_artifact = wandb.Artifact(name="training_code", type="code")
code_artifact.add_file(script_name)

# --- Define number of epochs ---
max_epochs = hyperparameters["max_epochs"]

# --- Scheduler Hyperparameters ---
lr_scheduler_patience = hyperparameters["lr_scheduler_patience"]
lr_scheduler_factor = hyperparameters["lr_scheduler_factor"]
min_lr = hyperparameters["min_lr"]
#max_lr_reductions = hyperparameters["max_lr_reductions"]
scheduler_mode = hyperparameters["lr_scheduler_mode"]

#optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) # torch optimizer
lr_rate = hyperparameters["learning_rate"] # Initial LR (Phase 1)
optimizer = bnb.optim.AdamW(
    model.parameters(),
    lr=lr_rate,
    betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
    weight_decay=hyperparameters["weight_decay"] # Initial WD (Phase 1)
) # paged optimizer

# --- Initialize Learning Rate Scheduler ---
scheduler = None # Will be initialized dynamically based on phase
current_scheduler_type = None # To track the current scheduler type

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
        "eval_every_n_steps": hyperparameters["eval_every_n_steps"],
        "initial_weight_decay": hyperparameters["weight_decay"], # Log initial WD
        "beta1": hyperparameters["beta1"],
        "beta2": hyperparameters["beta2"],
        "initial_grad_clip_norm": hyperparameters["grad_clip_norm"], # Log initial GCN
        "learning_rate_phases": hyperparameters["learning_rate_phases"], # Log phase LRs
        "weight_decay_phases": hyperparameters["weight_decay_phases"],   # Log phase WDs
        "grad_clip_norm_phases": hyperparameters["grad_clip_norm_phases"], # Log phase GCNs
        "phase_boundaries": hyperparameters["phase_boundaries"],          # Log phase boundaries

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
    }
    )

wandb.watch(model, log="all", log_freq=10)

start_time = time.time()
model.train()

# Initialize global best metric before the training loop
initialize_global_best_metric(eval_mode)

# --- Enable/Disable early stopping ---
early_stop_flag = hyperparameters["early_stop_flag"]
if early_stop_flag:
    # EarlyStopper now only needs patience.
    # The other parameters (delta, checkpoint_path, mode) will be used directly
    # by the save_checkpoint_if_improved function.
    early_stopper = EarlyStopping(
        patience=hyperparameters["early_stopping_patience"]
    )
    # The checkpoint_path for saving the best model is defined later and used directly.


# --- Training Loop ---

avg_train_loss = 0.0
eval_loss_real = 0.0
eval_loss_sim = 0.0
test_loss_real = 0.0
test_loss_sim = 0.0
global_step = 0  # Track total steps across all epochs
applied_phase_idx = -1 # Initialize applied phase index
steps_in_current_phase_for_eval_counter = 0 # Counter for phase-specific evaluation frequency

stop_training = False

last_eval_loss_real = last_sent_eval_acc_real = last_overall_eval_acc_real = last_eval_loss_sim = last_sent_eval_acc_sim = last_overall_eval_acc_sim = last_avg_eval_loss = last_avg_eval_acc = last_test_loss_real = last_sent_test_acc_real = last_overall_test_acc_real = last_test_loss_sim = last_sent_test_acc_sim = last_overall_test_acc_sim = last_avg_test_loss = last_avg_test_acc = 0.0 # initialize variables to store last known evaluation metrics, only for archival purposes. Global scope so can be used anywhere (mainly for logging)

for epoch in range(max_epochs): # epochs loop
    #total_epoch_train_loss = 0.0
    num_batches = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_dataloader): # batches loop

        # --- Phase Management ---
        current_phase_idx = 0
        if global_step >= hyperparameters["phase_boundaries"][1]: # Phase 3
            current_phase_idx = 2
        elif global_step >= hyperparameters["phase_boundaries"][0]: # Phase 2
            current_phase_idx = 1
        # Else: Phase 1 (current_phase_idx remains 0)

        if current_phase_idx != applied_phase_idx:
            new_lr = hyperparameters["learning_rate_phases"][current_phase_idx]
            new_wd = hyperparameters["weight_decay_phases"][current_phase_idx]
            new_gcn = hyperparameters["grad_clip_norm_phases"][current_phase_idx]
            new_scheduler_type = hyperparameters["lr_scheduler_types_phases"][current_phase_idx]
            new_keyword_weight = hyperparameters["keyword_weight_phases"][current_phase_idx]
            new_background_weight = hyperparameters["background_weight_phases"][current_phase_idx]
            new_eval_every_n_steps = hyperparameters["eval_every_n_steps_phases"][current_phase_idx]

            optimizer.param_groups[0]['lr'] = new_lr
            optimizer.param_groups[0]['weight_decay'] = new_wd
            hyperparameters["grad_clip_norm"] = new_gcn # Update for clipping logic
            hyperparameters["keyword_weight"] = new_keyword_weight # Update for loss calculation
            hyperparameters["background_weight"] = new_background_weight # Update for loss calculation
            hyperparameters["eval_every_n_steps"] = new_eval_every_n_steps # Update for eval frequency

            steps_in_current_phase_for_eval_counter = 0 # Reset eval counter for the new phase

            print(f"Global Step {global_step}: Entered Phase {current_phase_idx + 1}. "
                  f"LR: {new_lr}, WD: {new_wd}, Grad Clip: {new_gcn}, Scheduler: {new_scheduler_type}, "
                  f"Keyword Weight: {new_keyword_weight}, Background Weight: {new_background_weight}, "
                  f"Eval Every: {new_eval_every_n_steps} steps")
            wandb.log({
                "current_phase": current_phase_idx + 1,
                "phase_learning_rate": new_lr,
                "phase_weight_decay": new_wd,
                "phase_grad_clip_norm": new_gcn,
                "phase_scheduler_type": new_scheduler_type,
                "phase_keyword_weight": new_keyword_weight,
                "phase_background_weight": new_background_weight,
                "phase_eval_every_n_steps": new_eval_every_n_steps,
                "step": global_step,
                "epoch": epoch + 1
            })
            
            # --- Initialize or Re-initialize Scheduler for the new phase ---
            if new_scheduler_type != current_scheduler_type or scheduler is None:
                print(f"Initializing scheduler for Phase {current_phase_idx + 1}: {new_scheduler_type}")
                if new_scheduler_type == "ReduceLROnPlateau":
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode=hyperparameters["lr_scheduler_mode"],
                        factor=hyperparameters["lr_scheduler_factor"],
                        patience=hyperparameters["lr_scheduler_patience"],
                        verbose=True,
                        min_lr=hyperparameters["min_lr"]
                    )
                    lr_reduction_monitor = LRReductionMonitor()
                elif new_scheduler_type == "CosineAnnealingWarmRestarts":
                    t_0_current_phase = hyperparameters["cosine_warm_restarts_t_0_phases"][current_phase_idx]
                    t_mult_current_phase = hyperparameters["cosine_warm_restarts_t_mult_phases"][current_phase_idx]
                    scheduler = CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=t_0_current_phase,
                        T_mult=t_mult_current_phase,
                        eta_min=hyperparameters["min_lr"],
                        verbose=True,
                        last_epoch=-1
                    )
                elif new_scheduler_type == "StepLR":
                    step_size_current_phase = hyperparameters["step_lr_step_size_phases"][current_phase_idx]
                    gamma_current_phase = hyperparameters["step_lr_gamma_phases"][current_phase_idx]
                    scheduler = StepLR(
                        optimizer,
                        step_size=step_size_current_phase,
                        gamma=gamma_current_phase,
                        verbose=True
                    )
                else:
                    raise ValueError(f"Unsupported scheduler type: {new_scheduler_type} for phase {current_phase_idx + 1}")
                current_scheduler_type = new_scheduler_type
            
            applied_phase_idx = current_phase_idx
        # --- End Phase Management ---

        outputs, current_batch_size, labels = custom_forward(batch, model)
        logits  = outputs.logits 
        loss = weighted_loss_calculation(logits, labels, sentinel_ids, hyperparameters["keyword_weight"], hyperparameters["background_weight"])
                            
        #scaled_loss = loss / accum_steps                                   
        loss.backward()     

        log_memory_flex("After loss backward:",gpu_id=gpu_id, device_name=device_name)

        # Gradient Clipping
        if hyperparameters["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters["grad_clip_norm"])

        optimizer.step()
        optimizer.zero_grad()

        # --- Step LR Schedulers that are not ReduceLROnPlateau ---
        if current_scheduler_type in ["CosineAnnealingWarmRestarts", "StepLR"] and scheduler is not None:
            scheduler.step()

        log_memory_flex("After parameters update:",gpu_id=gpu_id, device_name=device_name)

        #total_epoch_train_loss += loss.item()*current_batch_size # loss.item() returns singular value of loss, also bring it back to CPU so can be mulitplied with current_batch_size
        total_samples += current_batch_size
        num_batches += 1
        global_step += 1
        steps_in_current_phase_for_eval_counter += 1

        # Log current batch loss
        print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Step {global_step}, Loss: {loss.item():.4f}")

        wandb.log({
            "train_loss": loss.item(),
            "epoch": epoch + 1,
            "step": global_step
        })

        del batch
        del outputs
        del loss
        torch.cuda.empty_cache()
        
        # Evaluate model every eval_every_n_steps (phase specific)
        if steps_in_current_phase_for_eval_counter % hyperparameters["eval_every_n_steps"] == 0:
            # Calculate average training loss for reporting
            #current_avg_train_loss = total_epoch_train_loss / total_samples if total_samples > 0 else 0
            #, Avg Train Loss: {current_avg_train_loss:.4f}
            
            #get eval loss and accuracy
            eval_loss_real, sent_eval_acc_real, overall_eval_acc_real = evaluate_model(model, eval_dataloader_real)
            
            print(f"Evaluation Real - Step {global_step}, Eval Loss Real: {eval_loss_real:.4f}, Eval Overall Accuracy Real: {overall_eval_acc_real:.4f}, Eval Key-tokens Accuracy Real: {sent_eval_acc_real:.4f}")

            eval_loss_sim, sent_eval_acc_sim, overall_eval_acc_sim = evaluate_model(model, eval_dataloader_sim)

            print(f"Evaluation Simulation - Step {global_step}, Eval Loss Sim: {eval_loss_sim:.4f},  Eval Overall Accuracy Sim: {overall_eval_acc_sim:.4f}, Eval Key-tokens Accuracy Sim: {sent_eval_acc_sim:.4f}")
            
            #calculate weighted average of eval loss
            avg_eval_loss = (eval_loss_real*hyperparameters["real_image_eval_weight"] + eval_loss_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            #calculate weighted average of eval accuracy
            avg_eval_acc = (overall_eval_acc_real*hyperparameters["real_image_eval_weight"] + overall_eval_acc_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            #get test loss and accuracy
            test_loss_real, sent_test_acc_real, overall_test_acc_real = evaluate_model(model, test_dataloader_real)

            print(f"Test Real - Step {global_step}, Test Loss Real: {test_loss_real:.4f}, Test Overall Accuracy Real: {overall_test_acc_real:.4f}, Test Key-tokens Accuracy Real: {sent_test_acc_real:.4f}")

            test_loss_sim, sent_test_acc_sim, overall_test_acc_sim = evaluate_model(model, test_dataloader_sim)

            print(f"Test Sim - Step {global_step}, Test Loss Sim: {test_loss_sim:.4f}, Test Overall Accuracy Sim: {overall_test_acc_sim:.4f}, Test Key-tokens Accuracy Sim: {sent_test_acc_sim:.4f}")

            #calculate weighted average of test loss
            avg_test_loss = (test_loss_real*hyperparameters["real_image_eval_weight"] + test_loss_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])
            
            #calculate weighted average test accuracy
            avg_test_acc = (overall_test_acc_real*hyperparameters["real_image_eval_weight"] + overall_test_acc_sim*hyperparameters["sim_image_eval_weight"]) / (hyperparameters["real_image_eval_weight"] + hyperparameters["sim_image_eval_weight"])

            # Log evaluation metrics
            wandb.log({
                "eval_loss_real": eval_loss_real,
                "eval_loss_sim": eval_loss_sim,
                "avg_eval_loss": avg_eval_loss,
                "test_loss_real": test_loss_real,
                "test_loss_sim": test_loss_sim,
                "avg_test_loss": avg_test_loss,
                "sent_eval_acc_real": sent_eval_acc_real,
                "sent_eval_acc_sim": sent_eval_acc_sim,
                "overall_eval_acc_real": overall_eval_acc_real,
                "overall_eval_acc_sim": overall_eval_acc_sim,
                "overall_test_acc_real": overall_test_acc_real,
                "overall_test_acc_sim": overall_test_acc_sim,
                "avg_test_acc": avg_test_acc,
                "current_lr": optimizer.param_groups[0]['lr'], # Log current learning rate
                "step": global_step
            })


            if eval_mode == "loss":
                evaluation_metric = avg_eval_loss
                print(f"Using Average Eval Loss ({avg_eval_loss:.4f}) for LR scheduling and early stopping.")
            elif eval_mode == "accuracy":
                evaluation_metric = avg_eval_acc
                print(f"Using Average Eval Accuracy ({avg_eval_acc:.4f}) for LR scheduling and early stopping.")
            else:
                # This check is also done when setting hyperparameters, but good for safety
                raise ValueError(f"Invalid eval_mode: {eval_mode}. Please choose 'loss' or 'accuracy'.")
            
            metric_improved_and_checkpoint_saved = save_checkpoint_if_improved(
                    current_metric_value=evaluation_metric,
                    model=model,
                    step=global_step,
                    epoch=epoch + 1,
                    checkpoint_path=best_checkpoint_path, # This is the existing path for best checkpoints
                    mode=eval_mode,
                    delta=hyperparameters["improvement_delta"]
                ) # Save checkpoint if improvement is met, returns Improved: True or False
            
            if current_scheduler_type == "ReduceLROnPlateau" and scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr'] # Get LR before potential reduction by scheduler
                scheduler.step(evaluation_metric) # Step the scheduler using the chosen evaluation_metric
                new_lr = optimizer.param_groups[0]['lr'] # Get LR after potential reduction

                lr_reduction_monitor.compare_lr(old_lr, new_lr)

            
            # --- Learning Rate Scheduler and Early Stopping Logic ---
            if early_stop_flag:
               
                # Check early stopping using the chosen evaluation_metric
                # The save_checkpoint_if_improved function now handles the logic of comparing with best_metric_value_global
                # and saving if improvement is met according to hyperparameters["early_stopping_delta"]
            
                early_stopper.check(metric_improved_and_checkpoint_saved) # reset internal counter if improved is true, otherwise increase by 1. If patience is met, set patience_met to True

                # Modified early stopping condition based on LR reductions and patience
                if early_stopper.patience_met:
                    if current_scheduler_type == "ReduceLROnPlateau" and scheduler is not None:
                        if lr_reduction_monitor.lr_reduction_count >= hyperparameters["max_lr_reductions"]:
                            print(f"Early stopping triggered: LR reduced {lr_reduction_monitor.lr_reduction_count}/{hyperparameters['max_lr_reductions']} times and {eval_mode} metric ({evaluation_metric:.4f}) hasn't improved for {early_stopper.patience} evaluations (patience met). Early stopping.")
                            stop_training = True
                            break
                        else:
                            print(f"{eval_mode.capitalize()} metric ({evaluation_metric:.4f}) hasn't improved for {early_stopper.patience} evaluations (patience met), but max LR reductions ({lr_reduction_monitor.lr_reduction_count}/{hyperparameters['max_lr_reductions']}) not reached yet. Current LR: {new_lr}. Continuing training.")
                            stop_training = False # redudant set to False, but good for clarity and safety
                    else:
                        new_lr = optimizer.param_groups[0]['lr']
                        print(f"{eval_mode.capitalize()} metric ({evaluation_metric:.4f}) hasn't improved for {early_stopper.patience} evaluations (patience met), scheduler is {current_scheduler_type}. Early stopping.")
                        stop_training = True
                        break
            
            last_eval_loss_real, last_sent_eval_acc_real, last_overall_eval_acc_real, last_eval_loss_sim, last_sent_eval_acc_sim, last_overall_eval_acc_sim, last_avg_eval_loss, last_test_loss_real, last_sent_test_acc_real, last_overall_test_acc_real, last_test_loss_sim, last_sent_test_acc_sim, last_overall_test_acc_sim, last_avg_test_loss, last_avg_test_acc = eval_loss_real, sent_eval_acc_real, overall_eval_acc_real, eval_loss_sim, sent_eval_acc_sim, overall_eval_acc_sim, avg_eval_loss, test_loss_real, sent_test_acc_real, overall_test_acc_real, test_loss_sim, sent_test_acc_sim, overall_test_acc_sim, avg_test_loss, avg_test_acc # update last known evaluation metrics, could have created a list for this and changes them via the list but whatever


    # If early stopping was triggered, break out of the epoch loop too
    if early_stop_flag and stop_training:
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
        print(f"first epoch checkpoint saved at time:{time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Add metadata about this checkpoint
        metadata = {
            "eval_mode": eval_mode,
            "epoch": epoch+1,
            "step": global_step,
            #"train_loss": total_epoch_train_loss / total_samples if total_samples > 0 else 0.0, # Use current epoch's avg train loss
            "eval_metric_real": last_eval_loss_real if eval_mode == "loss" else last_overall_eval_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "eval_metric_sim": last_eval_loss_sim if eval_mode == "loss" else last_overall_eval_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "avg_eval_metric": last_avg_eval_loss if eval_mode == "loss" else last_avg_eval_acc,
            "test_metric_real": last_test_loss_real if eval_mode == "loss" else last_overall_test_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "test_metric_sim": last_test_loss_sim if eval_mode == "loss" else last_overall_test_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "avg_test_metric": last_avg_test_loss if eval_mode == "loss" else last_avg_test_acc,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(first_epoch_chkpoint, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        wandb.log({"first_epoch_checkpoint_saved": True})

# --- Log artifacts to wandb ---
run.log_artifact(dataset_artifact)
run.log_artifact(code_artifact)

# Save the fine-tuned model at the end if early stopping wasn't used
if not early_stop_flag:
    final_checkpoint_path = os.path.join(hyperparameters["checkpoint_dir"], "final_save/")
    model.save_pretrained(final_checkpoint_path)
    
    # Save metadata about the final checkpoint
    # Use last known average train loss from an eval step, or N/A if no eval step occurred.
    #final_train_loss = last_train_loss if 'last_train_loss' in locals() and global_step > 0 else "N/A"

    metadata = {
            "eval_mode": eval_mode,
            "epoch": epoch+1,
            "step": global_step,
            #"train_loss": total_epoch_train_loss / total_samples if total_samples > 0 else 0.0, # Use current epoch's avg train loss
            "eval_metric_real": last_eval_loss_real if eval_mode == "loss" else last_overall_eval_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "eval_metric_sim": last_eval_loss_sim if eval_mode == "loss" else last_overall_eval_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "avg_eval_metric": last_avg_eval_loss if eval_mode == "loss" else last_avg_eval_acc,
            "test_metric_real": last_test_loss_real if eval_mode == "loss" else last_overall_test_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "test_metric_sim": last_test_loss_sim if eval_mode == "loss" else last_overall_test_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "avg_test_metric": last_avg_test_loss if eval_mode == "loss" else last_avg_test_acc,
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
            "final_test_metric_real": last_test_loss_real if eval_mode == "loss" else last_overall_test_acc_real, # Will be 0.0 if no eval step yet, otherwise last eval
            "final_test_metric_sim": last_test_loss_sim if eval_mode == "loss" else last_overall_test_acc_sim,   # Will be 0.0 if no eval step yet, otherwise last eval
            "final_avg_test_metric": last_avg_test_loss if eval_mode == "loss" else last_avg_test_acc,
        })


wandb.finish()
