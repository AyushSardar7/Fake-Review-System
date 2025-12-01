import re
import pandas as pd

# -----------------------------
# 1. Simple text cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text)

    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # remove emojis / non-ASCII (crude but effective)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    # basic lengths
    df["review_length_tokens"] = df["review_text"].str.split().str.len()
    df["review_length_chars"] = df["review_text"].str.len()

    # punctuation patterns
    df["exclamation_count"] = df["review_text"].str.count("!")
    df["question_count"] = df["review_text"].str.count(r"\?")

    # proportion of uppercase characters (shouting / emphasis)
    def upper_ratio(text: str) -> float:
        if not text:
            return 0.0
        total = len(text)
        upper = sum(1 for c in text if c.isupper())
        return upper / total if total > 0 else 0.0

    df["upper_ratio"] = df["review_text"].apply(upper_ratio)

    return df


# -----------------------------
# 2. Load raw datasets
# -----------------------------
print("📥 Loading raw datasets...")

google = pd.read_csv("../../data/raw/google_play_ecommerce_reviews_limited.csv")


# -----------------------------
# 4. Clean Google Play dataset
# -----------------------------
google = google.rename(columns={"content": "review_text"})
google = google.dropna(subset=["review_text", "score"])
google["review_text"] = google["review_text"].astype(str).apply(clean_text)
google["platform"] = "GooglePlay"

# sentiment proxy label from stars (NOT true fake/genuine)
google["label"] = google["score"].apply(
    lambda x: 1 if x >= 4 else (0 if x <= 2 else None)
)
google = google.dropna(subset=["label"])
google["label"] = google["label"].astype(int)

# fill missing numeric metadata
if "thumbsUpCount" in google.columns:
    google["thumbsUpCount"] = google["thumbsUpCount"].fillna(0).astype(int)
else:
    google["thumbsUpCount"] = 0

# metadata-based flags
google["is_anonymous_user"] = google["userName"].fillna("").str.contains(
    "google user", case=False
).astype(int)

google["has_reply"] = google["replyContent"].fillna("").str.strip().ne("").astype(int)

# add text features
google = add_text_features(google)

# -----------------------------
# 5. Heuristic fake-suspicion score (ONLY GooglePlay)
#    This is NOT ground-truth, just a useful extra signal.
# -----------------------------
def heuristic_fake_score(row) -> int:
    score = 0

    # overly short 5-star or 1-star review
    if row["review_length_tokens"] <= 4 and row["score"] in [1, 5]:
        score += 1

    # no likes
    if row["thumbsUpCount"] == 0:
        score += 1

    # anonymous generic account
    if row["is_anonymous_user"] == 1:
        score += 1

    # heavy punctuation / shouting
    if row["exclamation_count"] >= 3 or row["upper_ratio"] > 0.25:
        score += 1

    return score


google["heuristic_fake_score"] = google.apply(heuristic_fake_score, axis=1)
google["heuristic_fake_flag"] = (google["heuristic_fake_score"] >= 3).astype(int)

print("Google Play samples (after filtering):", len(google))

# -----------------------------
# 6. Combine datasets
# -----------------------------
combined = pd.concat(
    [
        google[
            [
                "review_text",
                "label",
                "platform",
                "review_length_tokens",
                "review_length_chars",
                "exclamation_count",
                "question_count",
                "upper_ratio",
                "thumbsUpCount",
                "is_anonymous_user",
                "has_reply",
                "heuristic_fake_score",
                "heuristic_fake_flag",
            ]
        ],
    ],
    ignore_index=True,
)

combined = combined.sample(frac=1, random_state=42)  # shuffle

combined.to_csv(
    "../../data/processed/real_labeled_reviews_v2.csv",
    index=False,
    encoding="utf-8",
)

print("🎯 Final dataset created: real_labeled_reviews_v2.csv")
print("Total samples:", len(combined))
print(combined["platform"].value_counts())
print(combined.head())
