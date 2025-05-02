import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, fbeta_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from sklearn.utils.class_weight import compute_class_weight
import os
import random
from imblearn.under_sampling import RandomUnderSampler


# Configure Logger
def setup_logger(log_dir, full_model_title):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/{full_model_title}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


# Load dataset
def load_data(file_path, text_column, label_column):
    df = pd.read_csv(file_path)
    return df[text_column].tolist(), df[label_column].tolist()


# under sampling
def undersample_data(texts, labels):
    texts = np.array(texts).reshape(-1, 1)
    undersampler = RandomUnderSampler(sampling_strategy=1)  # Equalize both classes
    texts_resampled, labels_resampled = undersampler.fit_resample(texts, labels)
    texts_resampled = texts_resampled.flatten()
    return texts_resampled, labels_resampled


# Dataset Class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Training function
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label"].to(device),
        )

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
    f2 = fbeta_score(true_labels, predictions, beta=2, average='binary')

    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    return accuracy, precision, recall, f1, f2, cm


if __name__ == "__main__":

    # file paths
    pretrained_path = "models/roberta-pretrained3"
    # coaid_path = "datasets/coaid_df.csv"
    coaid_path = "datasets/constraint_df.csv"
    constraint_path = "datasets/coaid_df2.csv"
    politifact_path = "datasets/politi_df.csv"

    # Parameters
    # pretrained = False
    pretrained = True
    coaid_percent = 100
    max_length = 64
    batch_size = 16
    epochs = 10
    learning_rate = 3e-5
    test_split = 0.2
    layers_frozen = 0

    if pretrained:
        model_name = pretrained_path
        model_log = "roberta-pretrained3"
    else:
        model_name = "roberta-base"
        model_log = "roberta-base3"

    # seed = 40
    # seed = 41
    # seed = 42
    # seed = 43
    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    full_model_title = f"{model_log} fine-tuned on constraint - seed {seed}"
    results_file_path = f"models/{full_model_title}"

    # Logger
    logger = setup_logger(results_file_path, full_model_title)

    # Log Model & Parameters
    logger.info(f"{model_log} with {layers_frozen} layers frozen fine-tuned on {coaid_percent}% of coaid")
    logger.info(f"--------------------------------------------------------------------------------------")
    logger.info(f"Parameters: Max Length: {max_length}, Batch Size: {batch_size}, "
                f"Epochs: {epochs}, LR: {learning_rate}, Test Set: {test_split}")
    logger.info(f"--------------------------------------------------------------------------------------")

    # Load data
    coaid_texts, coaid_labels = load_data(coaid_path, "clean_text", "new_label")
    cons_texts, cons_labels = load_data(constraint_path, "clean_text", "label")
    polit_texts, politi_labels = load_data(politifact_path, "clean_text", "label")

    # Apply under-sampling to coaid data
    # coaid_texts, coaid_labels = undersample_data(coaid_texts, coaid_labels)

    # Split coaid data
    train_texts, test_texts, train_labels, test_labels = train_test_split(coaid_texts, coaid_labels,
                                                                          test_size=test_split)

    # Reduce training data for coaid
    if coaid_percent < 100:
        train_data = list(zip(train_texts, train_labels))
        train_data_sample = random.sample(train_data, int((coaid_percent / 100) * len(train_data)))
        # Unzip the reduced training data
        train_texts_reduced, train_labels_reduced = zip(*train_data_sample)
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts_reduced, train_labels_reduced,
                                                                            test_size=0.2)
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels,
                                                                            test_size=0.2)

    # Tokenizer & Model
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze layers
    if layers_frozen > 0:
        for param in model.roberta.encoder.layer[:layers_frozen].parameters():
            param.requires_grad = False

    # Create datasets
    dataset_train = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    dataset_val = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    dataset_test = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    dataset_cons = TextClassificationDataset(cons_texts, cons_labels, tokenizer, max_length=max_length)
    dataset_politi = TextClassificationDataset(polit_texts, politi_labels, tokenizer, max_length=max_length)

    # DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_cons = DataLoader(dataset_cons, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_politi = DataLoader(dataset_politi, batch_size=batch_size, shuffle=False, num_workers=4)

    # Compute class weights
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = len(dataloader_train) * epochs
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    best_val_f1 = 0
    patience = 3
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, dataloader_train, loss_fn, optimizer, device)
        acc, prec, rec, f1, f2, cm = evaluate(model, dataloader_val, device)
        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Validation: Acc: {acc:.4f},"
                    f" Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
        logger.info(f"Confusion Matrix: TP={cm[1, 1]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TN={cm[0, 0]}")
        logger.info(f"--------------------------------------------------------------------------------------")

        # Save the best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), f"{results_file_path}/best_roberta_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered!")
            break

        scheduler.step()

    logger.info("Training complete.")
    logger.info(f"--------------------------------------------------------------------------------------")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(f"{results_file_path}/best_roberta_model.pt"))

    acc, prec, rec, f1, f2, cm = evaluate(model, dataloader_test, device)
    logger.info(f"Testing on Coaid test set: Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
    logger.info(f"Confusion Matrix: TP={cm[1, 1]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TN={cm[0, 0]}")
    logger.info(f"--------------------------------------------------------------------------------------")

    acc, prec, rec, f1, f2, cm = evaluate(model, dataloader_cons, device)
    logger.info(f"Testing on Constraint: Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
    logger.info(f"Confusion Matrix: TP={cm[1, 1]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TN={cm[0, 0]}")
    logger.info(f"--------------------------------------------------------------------------------------")

    acc, prec, rec, f1, f2, cm = evaluate(model, dataloader_politi, device)
    logger.info(f"Testing on Politifact: Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
    logger.info(f"Confusion Matrix: TP={cm[1, 1]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TN={cm[0, 0]}")
    logger.info(f"--------------------------------------------------------------------------------------")

    # Save final model
    model.save_pretrained(results_file_path)
    tokenizer.save_pretrained(results_file_path)
    logger.info("Model saved!")
