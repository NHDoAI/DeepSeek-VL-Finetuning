import json
import os
import random
from dotenv import load_dotenv
import time
import subprocess

import torch
from torch.utils.data import DataLoader, Dataset
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from peft import get_peft_model, LoraConfig
from deepseek_vl.utils.io import load_pil_images

import numpy as np

import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model_type = "1.3B"

checkpoint_dir = "./1.3b_less-lora_rerun_v7_deterministic_seed/"


# ------ Define useful functions and classes ------

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

# --- Function to evaluate the model's performance ---
def evaluate_model(model, eval_dataloader, device):
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
    
    with torch.no_grad():
        for batch in eval_dataloader:
            current_batch_size = batch["input_ids"].shape[0]
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'pixel_values' in batch:
                batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
            inputs_embeds = model.prepare_inputs_embeds(**batch)
        
            del batch["pixel_values"]
            labels = batch["input_ids"].clone()
            # For each sequence in the batch, mask out the first 1014 tokens
            labels[:, 0:1014] = -100
            batch["labels"] = labels

            # Forward pass through language model
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask'],
                labels= batch["labels"]  
            )
            
            loss = outputs.loss
            total_loss += loss.item()*current_batch_size
            total_samples += current_batch_size
            del inputs_embeds
            del batch
            del outputs
            del loss
            torch.cuda.empty_cache()

    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    model.train()
    return avg_loss

# --- Class for early stopping ---
class EarlyStopping: #fix according to chatgpt
    def __init__(self, patience=3, delta=0, checkpoint_path="./best_model"):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.best_epoch = -1

    def save_checkpoint(self, model, epoch):
        #save model when eval loss drops
        model.save_pretrained(self.checkpoint_path)
        self.best_epoch = epoch
        
        # Save metadata with epoch information
        metadata = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        # Save metadata to a JSON file
        with open(os.path.join(self.checkpoint_path, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        # Log to wandb
        #wandb.log({"best_model_epoch": epoch, "best_model_loss": self.best_loss})
        
        print(f"Earlystop Model saved to {self.checkpoint_path} at epoch {self.best_epoch}")

    def check(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        

# 1) pick ONE master seed for the whole run
MASTER_SEED = 42          # change this between experiments

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
    project="deepseek-vl-training",  # name of your project in wandb
    name="1.3b_less-lora_cluster_b6_re-run_v7_deterministic_seed",            # optional: custom run name
)

# --- Load environment variables ---
load_dotenv(override=True)


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



# TODO: test Lora and how to apply to different layers, how to freeze some module
# --- Apply LoRA Adapters to the Modules via configs ---

lora_rank = 6
lora_alpha = 12
lora_dropout = 0.15

if model_type == "1.3B":
    # --- 1.3b model ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "attn.qkv", "attn.proj", "fc1", "fc2","patch_embed.proj", "attn_pool.q", "attn_pool.kv",
                        #"aligner.layers.0", "aligner.layers.2"
                        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
elif model_type == "7B":
    # --- 7b model ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", #language model
                        "gate_proj", "up_proj", "down_proj", #language model
                        "attn.qkv", "attn.proj", "fc1", "fc2","patch_embed.proj", "attn_pool.q", "attn_pool.kv", #vision model
                        #"aligner.high_up_proj", "aligner.low_up_proj", "aligner.layers.1"
                        ], #aligner
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

# --- Load Model with LoRA Adapters ---
model = get_peft_model(model, lora_config, adapter_name="adapter")

# --- Define checkpoint path for saving the model-checkpoints ---

best_checkpoint_path = "saved_model_1.3b_less-lora_cluster_b6_re-run_v7_deterministic_seed/"
checkpoint_path = checkpoint_dir+best_checkpoint_path



# --- Prepare Training Loop ---
# TODO: check what are all the hyperparameters to be defined and group them into a config file

# --- Prepare Datasets ---

train_labels_dir = os.getenv('TRAIN_LABELS_PATH')
print(train_labels_dir)
eval_labels_dir = os.getenv('EVAL_LABELS_PATH')
print(eval_labels_dir)
test_labels_dir = os.getenv('TEST_LABELS_PATH') #to be changed for cluster path
print(test_labels_dir)

# Split data into train and eval sets
#train_files, eval_files = split_data_files(labels_dir, eval_ratio=0.2)

train_files = [os.path.join(train_labels_dir, fname) for fname in os.listdir(train_labels_dir) if fname.endswith('.json')]
eval_files = [os.path.join(eval_labels_dir, fname) for fname in os.listdir(eval_labels_dir) if fname.endswith('.json')]
test_files = [os.path.join(test_labels_dir, fname) for fname in os.listdir(test_labels_dir) if fname.endswith('.json')]

# Create train and eval datasets
train_dataset = MyMultimodalDataset(data_files=train_files, chat_processor=vl_chat_processor)
train_sampler  = torch.utils.data.RandomSampler(train_dataset,
                                               generator=g,
                                               replacement=False)

eval_dataset = MyMultimodalDataset(data_files=eval_files, chat_processor=vl_chat_processor)
test_dataset = MyMultimodalDataset(data_files=test_files, chat_processor=vl_chat_processor)

# --- Define hyperparameters ---

batch_size = 6

# Before the loop - only create dataloader references but not the actual dataset yet
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    generator=g,            # << this is what makes shuffling deterministic
    collate_fn=lambda batch: custom_collate_fn(
        batch,
        pad_keys=["input_ids", "attention_mask", "images_seq_mask"],
        tokenizer=tokenizer),
    pin_memory=True,
    persistent_workers=True,
)

eval_dataloader = DataLoader(
    eval_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
)

# --- Add artifacts to wandb ---
dataset_artifact = wandb.Artifact(name="test_data_artifact", type="dataset")
dataset_artifact.add_dir("./train/images")
dataset_artifact.add_dir("./train/labels")
dataset_artifact.add_dir("./eval/images")
dataset_artifact.add_dir("./eval/labels")
dataset_artifact.add_dir("./test/images")
dataset_artifact.add_dir("./test/labels")

code_artifact = wandb.Artifact(name="training_code", type="code")
code_artifact.add_file("training_v7.1_less-lora-rerun_1-3b_same-seed.py")

# --- Define number of epochs ---
num_epochs = 25

#optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) # torch optimizer
lr_rate = 2e-4
optimizer = bnb.optim.AdamW(model.parameters(), lr=lr_rate) # paged optimizer

wandb.config.update(
    {
        # optional: hyperparameter logging
        "run_name": run.name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr_rate,
        "optimizer": "AdamW (bnb)",
        "quantization": "4-bit",
        "lora_r": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout
    }
    )

wandb.watch(model, log="all", log_freq=10)

start_time = time.time()
model.train()

# --- Enable/Disable early stopping ---
early_stop_flag = True #TODO:modify to use argparse later

if early_stop_flag:
    early_stopper = EarlyStopping(patience=3, delta=0.0001, checkpoint_path=checkpoint_path)


# --- Training Loop ---

avg_train_loss = 0.0
eval_loss = 0.0
test_loss = 0.0

for epoch in range(num_epochs): # epochs loop
    total_train_loss = 0.0
    num_batches = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_dataloader): # batches loop

        current_batch_size = batch["input_ids"].shape[0]

        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if 'pixel_values' in batch:
            batch['pixel_values'] = batch['pixel_values'].to(torch.float16)

        # Prepare input embeddings using the model's method
        inputs_embeds = model.prepare_inputs_embeds(**batch)
        del batch["pixel_values"]

        labels = batch["input_ids"].clone()
        # For each sequence in the batch, mask out the first 1014 tokens
        labels[:, 0:1014] = -100
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
        batch["labels"] = labels
    

        #batch["labels"].to(model.device)
        
        # Convert pixel_values to float16 as in your inference code

        # Forward pass through language model
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['attention_mask'],
            labels= batch["labels"]  
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()*current_batch_size
        total_samples += current_batch_size
        num_batches += 1

        # Log current batch loss
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        wandb.log({
            "train_loss": loss.item(),
            "epoch": epoch + 1,
            "step": epoch * len(train_dataloader) + batch_idx + 1
        })
        del inputs_embeds
        del batch
        del outputs
        del loss
        torch.cuda.empty_cache()


    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} completed - Time: {epoch_time:.2f} seconds")
   

    wandb.log({
        "epoch_time": epoch_time,
        })
    
    avg_train_loss = total_train_loss / total_samples if total_samples > 0 else 0 # avg epoch loss

    # --- Evaluate model on eval set ---
    eval_loss = evaluate_model(model, eval_dataloader, model.device)
    print(f"Evaluation - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Eval Loss: {eval_loss:.4f}")

    wandb.log({ 
    "eval_loss": eval_loss,
    "epoch": epoch + 1,
    "step": epoch * len(train_dataloader) + batch_idx + 1
        })

    # --- Evaluate model on test set ---
    test_loss = evaluate_model(model, test_dataloader, model.device)
    print(f"Test - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Test Loss: {test_loss:.4f}")

    wandb.log({ 
    "test_loss": test_loss,
    "epoch": epoch + 1,
    "step": epoch * len(train_dataloader) + batch_idx + 1
        })
    if epoch == 0:
        first_epoch_chkpoint = checkpoint_dir+"first_epoch_chkpoint/"
        model.save_pretrained(first_epoch_chkpoint)
        print(f"first epoch checkpoint saved at time:{time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Add metadata about this checkpoint
        metadata = {
            "epoch": epoch+1,
            "train_loss": total_train_loss / total_samples,
            "eval_loss": eval_loss,
            "test_loss": test_loss,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(first_epoch_chkpoint, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        wandb.log({"first_epoch_checkpoint_saved": True})

    if early_stop_flag:
        if early_stopper.check(eval_loss):
            early_stopper.save_checkpoint(model=model, epoch=epoch+1)
        else:
            if early_stopper.early_stop:
                print("eval stagnation exceeded patience, early stopping activated")
                break

# --- Log artifacts to wandb ---
run.log_artifact(dataset_artifact)
run.log_artifact(code_artifact)

# Save the fine-tuned model at the end if early stopping wasn't used
if not early_stop_flag:
    final_checkpoint_path = checkpoint_dir+"final_save/"
    model.save_pretrained(final_checkpoint_path)
    
    # Save metadata about the final checkpoint
    metadata = {
        "final_epoch": num_epochs,
        "train_loss": avg_train_loss,
        "eval_loss": eval_loss,
        "test_loss": test_loss,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(final_checkpoint_path, "checkpoint_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    wandb.log({"final_model_epoch": num_epochs, "final_model_loss": eval_loss})

wandb.finish()
