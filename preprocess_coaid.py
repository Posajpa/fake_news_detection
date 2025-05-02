import pandas as pd
import re
import os


def merge_coaid(folder_path):
    fake_csv_files = [f for f in os.listdir(folder_path) if f.startswith("NewsFakeCOVID-19")]
    real_csv_files = [f for f in os.listdir(folder_path) if f.startswith("NewsRealCOVID-19")]

    fake_data_frame = []
    real_data_frame = []

    for file in fake_csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df["label"] = "1" # Add labels: 1 for fake
        fake_data_frame.append(df)

    for file in real_csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df["label"] = "0" # Add labels: 0 for real
        real_data_frame.append(df)

    fake_df = pd.concat(fake_data_frame, ignore_index=True)
    real_df = pd.concat(real_data_frame, ignore_index=True)
    merged_df = pd.concat([fake_df, real_df], ignore_index=True)
    return merged_df


def filter_coiad_df(df):
    filtered_df = df[df['type'] == "article"]
    return filtered_df


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def clean_dataframe(df):
    unwanted_phrases = [
        '403 forbidden', '404 error', '405 you have been blacklisted',
        '410 account suspended', 'access denied', 'not acceptable',
        'pagina non trovata', 'transcription editor', 'page not found'
    ]

    pattern = '|'.join(unwanted_phrases.lower() for unwanted_phrases in unwanted_phrases)
    df_clean = df[~df['newstitle'].str.lower().str.contains(pattern, na=False)].copy()

    # Remove rows with empty or quote-only titles explicitly
    df_clean['newstitle'] = df_clean['newstitle'].str.replace(r'^"+|"+$', '', regex=True).str.strip()
    df_clean = df_clean[df_clean['newstitle'].str.len() > 0]

    df_clean["clean_text"] = df_clean["newstitle"].apply(clean_text).str.strip()

    # Remove rows that become empty after cleaning
    df_clean = df_clean[df_clean["clean_text"].astype(bool)]

    return df_clean.reset_index(drop=True)


def clean_coiad_df(df):
    df = clean_dataframe(df)
    df.dropna(subset=["newstitle"], inplace=True)
    df["clean_text"] = df["newstitle"].apply(clean_text)
    df["clean_text"] = df["clean_text"].str.strip()
    df.replace({"clean_text": ""}, pd.NA, inplace=True)
    df.dropna(subset=["clean_text"], inplace=True)
    return df


def main_pipeline():
    coaid_df = merge_coaid('data_raw/coaid')
    print(len(coaid_df))
    filtered_df = filter_coiad_df(coaid_df)
    cleaned_df = clean_coiad_df(filtered_df)
    print(len(cleaned_df))
    cleaned_df.to_csv("datasets/coaid_df2.csv")


if __name__ == "__main__":
    main_pipeline()
