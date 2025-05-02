import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForMaskedLM, AdamW, get_scheduler, DataCollatorForLanguageModeling
)
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging
import os


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/pretraining.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


def load_data(file_path, text_column):
    df = pd.read_csv(file_path)
    return df[text_column].tolist()


# **Step 2: Tokenization**
class MaskedLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}  # Remove batch dim


# **Step 3: Training Function for MLM**
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    scaler = GradScaler()

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        with autocast():
            outputs = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                            labels=batch["input_ids"].to(device))
            loss = outputs.loss

        scaler.scale(loss).backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    # Parameters
    # model_name = "roberta-base"
    model_name = "models/roberta-pretrained"
    max_length = 64
    mlm_prob = 0.15
    batch_size = 32
    epochs = 50
    learning_rate = 1e-5
    data_file_path = "datasets/gdelt_df.csv"
    results_file_path = "models/roberta-pretrained3"

    # Logger
    logger = setup_logger(results_file_path)

    # Log Parameters
    logger.info(
        f"Model: {model_name}, Max Length: {max_length}, MLM Probability: {mlm_prob}, Batch Size: {batch_size}, "
        f"Epochs: {epochs}, LR: {learning_rate}")

    # Load data
    texts = load_data(data_file_path, "clean_text")

    tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=True)
    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # for param in model.roberta.encoder.layer[:4].parameters():
    #     param.requires_grad = False  # Optional: Freeze first 4 layers after a few epochs

    # Create Dataset
    dataset = MaskedLMDataset(texts, tokenizer, max_length)

    # Use DataCollator for MLM (Auto-Masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob, return_tensors="pt"
    )

    # Create DataLoader
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=data_collator)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = len(dataloader_train) * epochs
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    best_loss = float("inf")
    patience = 3
    patience_counter = 0

    # Training Loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}")

        train_loss = train(model, dataloader_train, optimizer, device)

        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")

        if epoch % 5 == 0:  # Save every 5 epochs
            torch.save(model.state_dict(), f"{results_file_path}/checkpoint_epoch_{epoch}.pt")

        # Save the best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"{results_file_path}/best_roberta_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered!")
            break

        scheduler.step()

    logger.info("Training complete.")

    # Save final model
    model.save_pretrained(results_file_path)
    tokenizer.save_pretrained(results_file_path)
    logger.info("Model saved!")

