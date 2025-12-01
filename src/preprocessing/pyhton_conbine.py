import pandas as pd

print("📥 Loading raw datasets...")

# Load datasets
amazon = pd.read_csv("../../data/processed/amazon_labeled_reviews.csv")
google = pd.read_csv("../../data/processed/real_labeled_reviews_v2.csv")  # contains heuristic flags

# -------------------------------
# Clean Amazon dataset
# -------------------------------
# Detect review_text column if needed
if "review_text" not in amazon.columns:
    for col in amazon.columns:
        if "review" in col.lower():
            amazon = amazon.rename(columns={col: "review_text"})
            break

amazon = amazon.dropna(subset=["review_text", "label"])
amazon["review_text"] = amazon["review_text"].astype(str)
amazon["platform"] = "Amazon"
amazon["label"] = amazon["label"].astype(int)

# Select exactly 60,000 reviews from Amazon for balanced dataset
amazon = amazon.sample(n=60000, random_state=42)

print("Amazon samples used:", len(amazon))

# -------------------------------
# Clean Google dataset (use heuristic_fake_flag as new label)
# -------------------------------
google = google.dropna(subset=["review_text", "heuristic_fake_flag"])
google["review_text"] = google["review_text"].astype(str)
google["platform"] = "GooglePlay"
google["label"] = google["heuristic_fake_flag"].astype(int)

google = google[["review_text", "label", "platform"]]
amazon = amazon[["review_text", "label", "platform"]]

print("Google samples:", len(google))

# -------------------------------
# Combine datasets
# -------------------------------
combined = pd.concat([amazon, google], ignore_index=True)
combined = combined.sample(frac=1, random_state=42)  # shuffle

combined.to_csv("../../data/processed/real_labeled_reviews.csv", index=False, encoding="utf-8")

print("🎯 Final dataset created: real_labeled_reviews.csv")
print("Total samples:", len(combined))
print(combined['platform'].value_counts())
print(combined.head())
