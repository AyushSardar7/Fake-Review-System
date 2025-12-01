import pandas as pd

def load_fasttext_file(file_path):
    reviews = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)  # split label and review
                labels.append(parts[0].replace("__label__", ""))  # 0 or 1
                reviews.append(parts[1])
    df = pd.DataFrame({"review_text": reviews, "label": labels})
    return df

# Load train and test
df_train = load_fasttext_file("train.ft.txt")
df_test = load_fasttext_file("test.ft.txt")

# Combine
df_amazon = pd.concat([df_train, df_test], ignore_index=True)
df_amazon["platform"] = "Amazon"

# Save to CSV
df_amazon.to_csv("data/raw/amazon_reviews.csv", index=False)
print("✅ Amazon reviews saved as CSV")
print(df_amazon.sample(5))
