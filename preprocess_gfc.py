import requests
import pandas as pd
import re
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


def fetch_gfc(api_key, base_url, queries, max_age_day=1, max_results=100):
    all_claims = []

    for query in queries:
        params = {
            'query': query,
            'key': api_key,
            "maxAgeDays": max_age_day,
            'pageSize': max_results,
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            claims = response.json().get("claims", [])
            all_claims.extend(claims)
        else:
            print(f"Error fetching claims for '{query}': {response.status_code}, {response.text}")

    return all_claims


def create_gfc_df(claims):
    data = []
    for claim in claims:
        claim_text = claim.get("text", "")
        claim_claimant = claim.get("claimant", "")
        claim_date = claim.get("claimDate", "")
        for review in claim.get("claimReview", []):
            publisher_name = review["publisher"].get("name", "")
            publisher_site = review["publisher"].get("site", "")
            review_url = review.get("url", "")
            review_title = review.get("title", "")
            review_date = review.get("reviewDate", "")
            textual_rating = review.get("textualRating", "")
            language_code = review.get("languageCode", "")
            data.append([
                claim_text,
                claim_claimant,
                claim_date,
                publisher_name,
                publisher_site,
                review_url,
                review_title,
                review_date,
                textual_rating,
                language_code
            ])

    df = pd.DataFrame(data, columns=[
        "claim_text",
        "claim_claimant",
        "claim_date",
        "publisher_name",
        "publisher_site",
        "review_url",
        "review_title",
        "review_date",
        "textual_rating",
        "language_code"
    ])
    return df


def filter_gfc_df(df):
    filtered_df = df[df['language_code'] == "en"]
    filtered_df['review_date'] = pd.to_datetime(filtered_df['review_date'])  # Convert to datetime
    filtered_df = filtered_df[filtered_df['review_date'].dt.year == 2020]  # Filter by year
    return filtered_df


def map_textual_rating_to_score(publisher_name, textual_rating):
    if publisher_name == "ABC":
        if "Misleading" in textual_rating:
            return "Partly False"
        elif "Yes, but more to it" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "AFP Fact Check":
        if "FALSE" in textual_rating:
            return "False"
        elif "Misleading" in textual_rating:
            return "Partly False"
        elif "Partly false" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "Africa Check":
        if "Mostly correct" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "Alt News":
        if "FALSE" in textual_rating:
            return "False"
        elif "Half True" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "AP News":
        if "Partly false" in textual_rating:
            return "Partly False"
        elif "False" in textual_rating:
            return "False"
        elif "Misleading" in textual_rating:
            return "Partly False"
        elif "Missing context" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "Australian Associated Press":
        if "Partly False" in textual_rating:
            return "Partly False"
        elif "False" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "BBC":
        if "not correct" in textual_rating:
            return "False"
        elif "FALSE" in textual_rating:
            return "False"
        elif "False" in textual_rating:
            return "False"
        elif "requires context" in textual_rating:
            return "Partly False"
        elif "incorrect" in textual_rating:
            return "False"
        elif "not correct" in textual_rating:
            return "False"
        elif "Fake" in textual_rating:
            return "False"
        elif "false" in textual_rating:
            return "False"
        elif "is fake" in textual_rating:
            return "False"
        elif "are fake" in textual_rating:
            return "False"
        elif "True" in textual_rating:
            return "True"
        elif "True" in textual_rating:
            return "True"
        elif "accurate" in textual_rating:
            return "True"
        elif "misleading" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "BOOM Fact Check":
        if "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "CBS 8":
        if "True" in textual_rating:
            return "True"
        else:
            return "Unknown"

    if publisher_name == "CBS News":
        if "False" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "FactCheck.org":
        if "FALSE" in textual_rating:
            return "False"
        elif "Misleading" in textual_rating:
            return "Partly False"
        elif "Distorts the Facts" in textual_rating:
            return "Partly False"
        elif "Is Flawed" in textual_rating:
            return "Partly False"
        elif "Disputed" in textual_rating:
            return "False"
        elif "Spins the Facts" in textual_rating:
            return "Partly False"
        elif "Not What Data Show" in textual_rating:
            return "Partly False"
        elif "Not the Whole Story" in textual_rating:
            return "Partly False"
        elif "Exaggerates" in textual_rating:
            return "Partly False"
        elif "Lacks Context" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "FACTLY":
        if "FALSE" in textual_rating:
            return "False"
        elif "MISLEADING" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "Full Fact":
        if "is correct" in textual_rating:
            return "True"
        elif "is accurate" in textual_rating:
            return "True"
        elif "Broadly correct" in textual_rating:
            return "True"
        elif "False." in textual_rating:
            return "False"
        elif "Incorrect." in textual_rating:
            return "False"
        elif "It has not." in textual_rating:
            return "False"
        elif "is incorrect" in textual_rating:
            return "False"
        elif "It is not." in textual_rating:
            return "False"
        elif "is true" in textual_rating:
            return "True"
        elif "It does," in textual_rating:
            return "True"
        elif "Partly False." in textual_rating:
            return "Partly False"
        elif "did not show." in textual_rating:
            return "Partly False"
        elif "out of context" in textual_rating:
            return "Partly False"
        elif "false rumour" in textual_rating:
            return "False"
        elif "untrue" in textual_rating:
            return "False"
        elif "is false" in textual_rating:
            return "False"
        elif "is misleading" in textual_rating:
            return "Partly False"
        elif "not accurate" in textual_rating:
            return "Partly False"
        elif "not correct" in textual_rating:
            return "Partly False"
        elif "is wrong" in textual_rating:
            return "False"
        elif "True" in textual_rating:
            return "True"
        else:
            return "Unknown"

    if publisher_name == "India Today":
        if "FALSE" in textual_rating:
            return "False"
        if "Mostly false" in textual_rating:
            return "False"
        elif "Half true" in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "KGW":
        if "True" in textual_rating:
            return "True"
        else:
            return "Unknown"

    if publisher_name == "KHOU":
        if "True" in textual_rating:
            return "True"
        else:
            return "Unknown"

    if publisher_name == "KVUE":
        if "False" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "localmemphis.com":
        if "True" in textual_rating:
            return "True"
        else:
            return "Unknown"

    if publisher_name == "News Center Maine":
        if "True" in textual_rating:
            return "True"
        elif "False" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "Newschecker":
        if "False Connection" in textual_rating:
            return "Partly False"
        elif "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "NewsMeter":
        if "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "PA Media":
        if "Misleading." in textual_rating:
            return "Partly False"
        else:
            return "Unknown"

    if publisher_name == "PolitiFact":
        if "FALSE" in textual_rating:
            return "False"
        elif "Pants on Fire" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "Science Feedback":
        if "Inaccurate" in textual_rating:
            return "False"
        elif "Incorrect" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "FALSE":
        if "Inaccurate" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "The Conversation":
        if "Not true" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "The New York Times":
        if "misleading" in textual_rating:
            return "False"
        elif "False" in textual_rating:
            return "False"
        elif "This is exaggerated." in textual_rating:
            return "Partly False"
        elif "FALSE" in textual_rating:
            return "False"
        elif "This is disputed." in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "The Quint":
        if "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "The Washington Post":
        if "Four Pinocchios" in textual_rating:
            return "False"
        elif "Bottomless Pinocchio" in textual_rating:
            return "Partly False"
        elif "Three Pinocchios" in textual_rating:
            return "Partly False"
        elif "Two Pinocchios." in textual_rating:
            return "Partly False"
        elif "One Pinocchio" in textual_rating:
            return "Partly False"
        elif "Wrong" in textual_rating:
            return "False"
        elif "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "USA Today":
        if "FALSE" in textual_rating:
            return "False"
        elif "Missing Context" in textual_rating:
            return "Partly False"
        elif "Partly False" in textual_rating:
            return "Partly False"
        elif "Partly True" in textual_rating:
            return "Partly False"
        elif "false" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "VERIFYThis.com":
        if "False." in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "VOA":
        if "FALSE" in textual_rating:
            return "False"
        elif "Misleading" in textual_rating:
            return "Partly False"
        elif "Partly False" in textual_rating:
            return "Partly False"
        elif "Dangerous" in textual_rating:
            return "    False"
        elif "Disputed" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "VOA News":
        if "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "Washington Post":
        if "Four Pinocchios" in textual_rating:
            return "False"
        elif "Bottomless Pinocchio" in textual_rating:
            return "Partly False"
        elif "Three Pinocchios" in textual_rating:
            return "Partly False"
        elif "Two Pinocchios." in textual_rating:
            return "Partly False"
        elif "One Pinocchio" in textual_rating:
            return "Partly False"
        elif "Wrong" in textual_rating:
            return "False"
        elif "FALSE" in textual_rating:
            return "False"
        else:
            return "Unknown"

    if publisher_name == "ZimFact":
        if "is false" in textual_rating:
            return "False"
        else:
            return "Unknown"

    elif "FALSE" in textual_rating:
        return "False"

    else:
        return "Unknown"


def label_verdict(verdict):
    verdict = verdict.lower()
    if "true" in verdict:
        return 0  # true
    elif "unknown" in verdict:
        return 2  # unknown
    else:
        return 1


def map_gfc_df(df):
    df["verdict_label"] = df.apply(
        lambda row: map_textual_rating_to_score(row["publisher_name"], row["textual_rating"]),
        axis=1
    )
    df["numerical_label"] = df["verdict_label"].apply(label_verdict)
    df = df[df["numerical_label"] != 2]
    return df

import re


def negate_sentence(sentence):
    # Simple negation rule: add "not" before the main verb
    words = sentence.split()
    for i, word in enumerate(words):
        if word.lower() in ["is", "was", "are", "were", "has", "had"]:
            words.insert(i + 1, "not")
            return " ".join(words)

    # If no verb is found, just prepend "It is not true that"
    return "It is not true that " + sentence


def create_true_gfc(df):
    # Select "False" labeled articles
    df_false = df[df["verdict_label"] == "False"].copy()

    # Apply the function to generate "True" versions
    df_false["claim_text"] = df_false["claim_text"].apply(negate_sentence)

    # Change the label to "True"
    df_false["verdict_label"] = "True"
    df_false["numerical_label"] = 0

    # Merge back with the original dataset
    df_balanced = pd.concat([df, df_false], ignore_index=True)

    return df_balanced


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def clean_gfc_df(df):
    df["clean_text"] = df["claim_text"].apply(clean_text)
    df["clean_text"] = df["clean_text"].str.strip()
    df.replace({"clean_text": ""}, pd.NA, inplace=True)
    df.dropna(subset=["clean_text"], inplace=True)
    return df


def main_pipeline():
    api_key = ""  # Replace with your actual API key
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    queries = ["covid", "coronavirus", "virus", "vaccine"]
    max_age_day = 1000000
    max_results = 1000000
    gfc_fd = fetch_gfc(api_key, base_url, queries, max_age_day, max_results)
    gfc_df = create_gfc_df(gfc_fd)
    filtered_dataframe = filter_gfc_df(gfc_df)
    mapped_df = map_gfc_df(filtered_dataframe)
    df_balanced = create_true_gfc(mapped_df)
    cleaned_df = clean_gfc_df(df_balanced)
    # cleaned_df = clean_gfc_df(mapped_df)
    cleaned_df.to_csv("gfc_df.csv")


if __name__ == "__main__":
    main_pipeline()
