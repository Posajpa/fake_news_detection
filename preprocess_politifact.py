import pandas as pd
import re


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def clean_coiad_df(df):
    df["clean_text"] = df["title"].apply(clean_text)
    df["clean_text"] = df["clean_text"].str.strip()
    df.replace({"clean_text": ""}, pd.NA, inplace=True)
    df.dropna(subset=["clean_text"], inplace=True)
    return df


def main_pipeline():
    fake_news = pd.read_csv("data_raw\politifact\politifact_fake.csv")
    print(len(fake_news))
    real_news = pd.read_csv("data_raw\politifact\politifact_real.csv")
    print(len(real_news))

    # Add labels: 1 for fake, 0 for real
    fake_news["label"] = "1"
    real_news["label"] = "0"

    politi_df = pd.concat([fake_news, real_news], ignore_index=False)
    print(len(politi_df))

    cleaned_df = clean_coiad_df(politi_df)
    cleaned_df.dropna(subset=["clean_text"], inplace=True)
    cleaned_df.to_csv("datasets/politi_df.csv")


if __name__ == "__main__":
    main_pipeline()
