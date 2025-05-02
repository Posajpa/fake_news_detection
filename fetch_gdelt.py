from gdeltdoc import GdeltDoc, Filters
import json
from datetime import datetime, timedelta
import os


def gdelt_fetcher(keyword, start_date, end_date, num_records):
    """Fetch articles from GDELT based on search filters."""
    gd = GdeltDoc()
    f = Filters(
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        num_records=num_records,
    )
    try:
        fetched_articles = gd.article_search(f)
        return fetched_articles
    except Exception as e:
        print(f"{e}")
        return None


def save_articles_to_json(articles, file_path):
    """Save or append articles to a JSON file while maintaining valid JSON structure."""
    if os.path.exists(file_path):
        # If file exists, load existing data and append new articles
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        # If file doesn't exist, start with an empty list
        existing_data = []

    # Append new articles to the existing data
    existing_data.extend(articles)

    # Write back the combined data to the JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Articles saved to {file_path}")


def run_article_extraction(keywords, num_records, start_date, end_date, output_dir):
    start_date_time = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_time = datetime.strptime(end_date, '%Y-%m-%d')

    while start_date_time < end_date_time:
        loop_end_date = start_date_time + timedelta(days=1)

        start_date_str = start_date_time.strftime('%Y-%m-%d')
        end_date_str = loop_end_date.strftime('%Y-%m-%d')

        try:
            fetched_articles = gdelt_fetcher(keywords, start_date_str, end_date_str, num_records)
            english_fetched_articles = fetched_articles[fetched_articles["language"] == "English"]
            # Save to CSV
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{start_date_str}.csv")
            english_fetched_articles.to_csv(file_path, index=False, encoding="utf-8")
            print(f"Saved {len(english_fetched_articles)} articles to {file_path}")

        except Exception as e:
            # Handle errors and exit if something goes wrong
            print(f"{e}")
            exit(1)

        # Move to the next day
        start_date_time = loop_end_date


def main():
    # file path
    file_path = "data_raw/gdelt"

    # parameters
    keywords = ["COVID", "coronavirus"]
    num_records = 250
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    run_article_extraction(keywords, num_records, start_date, end_date, file_path)


if __name__ == "__main__":
    main()
