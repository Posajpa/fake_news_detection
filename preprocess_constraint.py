import requests
import pandas as pd
import re
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


def label_verdict(label):
    if "real" in label:
        return 0
    else:
        return 1


def map_constraint_df(df):
    df["new_label"] = df["label"].apply(label_verdict)
    return df


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def clean_constraint_df(df):
    df.dropna(subset=["tweet"], inplace=True)
    df["clean_text"] = df["tweet"].apply(clean_text)
    df["clean_text"] = df["clean_text"].str.strip()
    df.replace({"clean_text": ""}, pd.NA, inplace=True)
    df.dropna(subset=["clean_text"], inplace=True)
    return df


def main_pipeline():
    constraint_df = pd.read_csv("data_raw\constraint\english_test_with_labels - Sheet1.csv")
    print(len(constraint_df))
    # mapped_df = map_constraint_df(constraint_df)
    # cleaned_df = clean_constraint_df(mapped_df)
    # cleaned_df.to_csv("datasets/constraint_df.csv")


if __name__ == "__main__":
    main_pipeline()
