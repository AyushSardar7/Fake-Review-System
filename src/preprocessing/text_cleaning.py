import pandas as pd
import re
import emoji
from bs4 import BeautifulSoup

def clean_text(text):
    text = str(text).lower()                                  # lowercase
    text = BeautifulSoup(text, "html.parser").get_text()      # remove HTML tags
    text = emoji.replace_emoji(text, replace='')              # remove emojis
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)               # remove symbols
    text = re.sub(r"\s+", " ", text).strip()                  # remove extra spaces
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)            # remove repeating words
    return text

# Load dataset
df = pd.read_csv("../../data/processed/real_labeled_reviews.csv")

# Clean
df["cleaned_review"] = df["review_text"].apply(clean_text)

# Remove empty rows after cleaning
df = df[df["cleaned_review"].str.len() > 3]

df.to_csv("../../data/processed/real_cleaned_labeled_reviews.csv", index=False)
print("🔥 Done! Clean dataset saved.")
print(df.head())
