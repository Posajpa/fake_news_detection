import pandas as pd
import re
import os


def merge_gdelt(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    data_frames = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    merged_df = pd.concat(data_frames, ignore_index=True)
    return merged_df


def clean_text(text):
    text = text.lower().strip()  # Convert to lowercase & remove leading/trailing spaces
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Keep only alphanumeric & punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text


def clean_gdelt_df(df):
    df = df.dropna(subset=["title"]).copy()  # Drop missing values safely
    df["clean_text"] = df["title"].apply(clean_text)
    # Drop rows with empty strings after cleaning
    df = df[df["clean_text"].str.len() > 0]
    # Remove rows with less than 3 words
    df = df[df["clean_text"].str.split().str.len() > 5]
    return df


def main_pipeline():
    gdelt_df = merge_gdelt("data_raw/gdelt")
    # print(len(gdelt_df))
    cleaned_df = clean_gdelt_df(gdelt_df)
    cleaned_df.to_csv("datasets/gdelt_df.csv")


if __name__ == "__main__":
    main_pipeline()
