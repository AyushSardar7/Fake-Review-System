import pandas as pd

# Load your cleaned dataset
df = pd.read_csv("../../data/processed/cleaned_reviews.csv")

# Add a simple label column (example logic — adjust as needed)
df['label'] = df['cleaned_review'].apply(lambda x: 1 if len(str(x)) > 50 else 0)

# Save only the columns you need
df[['cleaned_review', 'label']].to_csv("../../data/processed/labeled_reviews.csv", index=False)

print("✅ labeled_reviews.csv created successfully!")