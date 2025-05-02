import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import logging
import os


# Configure Logger
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/experiment.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


# Load dataset
def load_data(file_path, text_column, label_column):
    df = pd.read_csv(file_path)
    return df[text_column].tolist(), df[label_column].tolist()


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

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    # Parameters
    model_name = "Experiments/wp 4 100g 100c"
    max_length = 256
    batch_size = 32
    data_file_path = "constraint_df.csv"
    results_file_path = f"{model_name} test results"

    # Logger
    logger = setup_logger(results_file_path)

    # Log Parameters
    logger.info(
        f"Model: {model_name}, Max Length: {max_length}, Batch Size: {batch_size}")

    # Load data
    test_texts, test_labels = load_data(data_file_path, "clean_text", "new_label")

    # Tokenizer & Model
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create datasets
    dataset_test = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    # DataLoader
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    acc, prec, rec, f1 = evaluate(model, dataloader_test, device)
    logger.info(f"Testing: Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, "
                f"F1: {f1:.4f}")
    logger.info("Testing complete.")
