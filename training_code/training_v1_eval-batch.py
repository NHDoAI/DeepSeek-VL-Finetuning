import json
import os
import random
from dotenv import load_dotenv
import time

import torch
from torch.utils.data import DataLoader, Dataset
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from peft import get_peft_model, LoraConfig
from deepseek_vl.utils.io import load_pil_images

import wandb

run = wandb.init(
    project="deepseek-vl-training",  # name of your project in wandb
    name="test_early_stop_run2",            # optional: custom run name
    #mode="offline"
)

early_stop_flag = True #TODO:modify to use argparse later

load_dotenv()

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
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'pixel_values' in batch:
                batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
            inputs_embeds = model.prepare_inputs_embeds(**batch)
        
            batch_labels = torch.empty(0, dtype=batch["input_ids"].dtype).to(model.device)
            for in_id in batch["input_ids"]:
                labels_tensor = in_id.clone()
                labels_tensor[0:1014] = -100
                batch_labels = torch.cat([batch_labels, labels_tensor], dim=0)
            batch["labels"] = batch_labels

            
            
            # Convert pixel_values to float16 as in your inference code

            # Forward pass through language model
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask'],
                labels= batch["labels"]  
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    model.train()
    return avg_loss

class EarlyStopping: #fix according to chatgpt
    def __init__(self, patience=3, delta=0, checkpoint_path="./best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def save_checkpoint(self, model, batch_idx, epoch):
        #save model when eval loss drops
        model.save_pretrained(self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path} at epoch {epoch} and batch {batch_idx}")

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

model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
labels_dir = os.getenv('TRAIN_LABELS_PATH')
test_labels_dir = os.getenv('TEST_LABELS_PATH') #to be changed for cluster path

# --- Step 1: Prepare Your Dataset ---
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Split data into train and eval sets
train_files, eval_files = split_data_files(labels_dir, eval_ratio=0.2)
test_files = [os.path.join(test_labels_dir, fname) for fname in os.listdir(test_labels_dir) if fname.endswith('.json')]


# Create train and eval datasets
train_dataset = MyMultimodalDataset(data_files=train_files, chat_processor=vl_chat_processor)
eval_dataset = MyMultimodalDataset(data_files=eval_files, chat_processor=vl_chat_processor)
test_dataset = MyMultimodalDataset(data_files=test_files, chat_processor=vl_chat_processor)

# Create dataloaders
batch_size = 1
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=lambda batch: custom_collate_fn(batch, pad_keys=["input_ids", "attention_mask", "images_seq_mask"], tokenizer=tokenizer)
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

# --- Step 2: Define the Quantization Configuration ---
# 8 bit version
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # Alternatively, use load_in_8bit=True if preferred
#     llm_int8_skip_modules=["vision_model", "aligner"]  # Skip quantizing vision and aligner modules
# )

# 4 bit version
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Note: llm_int8_skip_modules is ignored in 4-bit mode.
    )

config = AutoConfig.from_pretrained(model_path)
config.pad_token_id = tokenizer.eos_token_id
config.eos_token_id = tokenizer.eos_token_id
config.bos_token_id = tokenizer.bos_token_id
# --- Step 3: Load the Pretrained Model with Quantization ---

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=quantization_config,
    config=config,
    device_map="auto"
)

# TODO: test Lora and how to apply to different layers, how to freeze some module
# --- Step 4: Apply LoRA Adapters to the Language Model ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "attn.qkv", "attn.proj", "fc1", "fc2","patch_embed.proj", "attn_pool.q", "attn_pool.kv",
                    "aligner.layers.0", "aligner.layers.2"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config, adapter_name="adapter")

# --- Step 5: Training Loop ---
# TODO: check what are all the hyperparameters to be defined and group them into a config file
#optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) # torch optimizer
optimizer = bnb.optim.AdamW(model.parameters(), lr=2e-4) # paged optimizer
num_epochs = 15
eval_every = 4  # Evaluate every 50 batches

wandb.config.update(
    {                     # optional: hyperparameter logging
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "eval_every": eval_every,
        "learning_rate": 2e-4,
        "optimizer": "AdamW (bnb)",
        "quantization": "4-bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1
    }

)

dataset_artifact = wandb.Artifact(name="test_data_artifact", type="dataset")
dataset_artifact.add_dir("./images")
dataset_artifact.add_dir("./labels")

wandb.watch(model, log="all", log_freq=4)

checkpoint_path = "./saved_model_testrun-2"

start_time = time.time()
model.train()


if early_stop_flag:
    early_stopper = EarlyStopping(patience=4, delta=0, checkpoint_path=checkpoint_path)

for epoch in range(num_epochs):
    total_train_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if 'pixel_values' in batch:
            batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
        # Prepare input embeddings using the model's method
        inputs_embeds = model.prepare_inputs_embeds(**batch)
        
        batch_labels = torch.empty(0, dtype=batch["input_ids"].dtype).to(model.device)
        for in_id in batch["input_ids"]:
            labels_tensor = in_id.clone()
            labels_tensor[0:1014] = -100
            batch_labels = torch.cat([batch_labels, labels_tensor], dim=0)
        batch["labels"] = batch_labels
        
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
        
        total_train_loss += loss.item()
        num_batches += 1
        
        # Log current batch loss
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        wandb.log({
            "train_loss": loss.item(),
            "epoch": epoch + 1,
            "step": epoch * len(train_dataloader) + batch_idx + 1
        })


        # Evaluate periodically
        # TODO: consider option to modify to evaluate every epoch instead
        if (batch_idx + 1) % eval_every == 0:
            eval_loss = evaluate_model(model, eval_dataloader, model.device)
            print(f"Evaluation - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Eval Loss: {eval_loss:.4f}")

            wandb.log({ 
            "eval_loss": eval_loss,
            "epoch": epoch + 1,
            "step": epoch * len(train_dataloader) + batch_idx + 1
            })

            test_loss = evaluate_model(model, test_dataloader, model.device)
            print(f"Test - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Test Loss: {test_loss:.4f}")

            wandb.log({ 
            "test_loss": test_loss,
            "epoch": epoch + 1,
            "step": epoch * len(train_dataloader) + batch_idx + 1
            })

            if early_stop_flag:
                if early_stopper.check(eval_loss):
                    early_stopper.save_checkpoint(model, batch_idx+1, epoch+1)
                else:
                    if early_stopper.early_stop:
                        print("eval stagnation exceeded patience, early stopping activated")
                        break
    

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} completed - Time: {epoch_time:.2f} seconds")

    wandb.log({
        "epoch_time": epoch_time,
        })
    
    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
    
    if early_stop_flag and early_stopper.early_stop:
        print("Training stopped early!")
        break

    # Evaluate at the end of each epoch
    eval_loss = evaluate_model(model, eval_dataloader, model.device)
    
    print(f"Epoch {epoch+1}/{num_epochs} completed - Avg Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    wandb.log({
        "eval_loss": eval_loss,
        "epoch": epoch + 1,
        "eval_step": epoch * len(train_dataloader) + batch_idx + 1
    })

    test_loss = evaluate_model(model, test_dataloader, model.device)

    print(f"Test - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Test Loss: {test_loss:.4f}")

    wandb.log({ 
    "test_loss": test_loss,
    "epoch": epoch + 1,
    "step": epoch * len(train_dataloader) + batch_idx + 1
    })

run.log_artifact(dataset_artifact)
# Save the fine-tuned model
if not early_stop_flag:
    model.save_pretrained("./fine_tuned_model_quantized")
wandb.finish()