import pandas as pd

# Load full Amazon dataset
df = pd.read_csv("../../data/raw/amazon_reviews.csv")

# Keep only necessary columns
df = df[["review_text", "label"]]

# Randomly sample 90k rows
df = df.sample(n=90000, random_state=42)

# Convert 1→0 and 2→1
df["label"] = df["label"].replace({1: 0, 2: 1})

# Save final labeled dataset
df.to_csv("../../data/processed/amazon_labeled_reviews.csv", index=False, encoding="utf-8")

print("✅ Amazon labeled dataset saved successfully!")
print(df.head())
print("Total rows:", len(df))
