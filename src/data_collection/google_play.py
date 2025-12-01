from google_play_scraper import reviews, Sort
import pandas as pd

apps = {
    "Flipkart": "com.flipkart.android",
    "Myntra": "com.myntra.android",
    "Meesho": "com.meesho.supply",
    "Ajio": "com.ril.ajio"
}

all_reviews = []

for app_name, package in apps.items():
    print(f"📥 Scraping {app_name} (up to 10,000 reviews)...")
    try:
        result, _ = reviews(
            package,
            lang="en",
            country="in",
            count=10000,          # Fetch up to 10k reviews
            sort=Sort.NEWEST      # ✅ use enum instead of string
        )
        df = pd.DataFrame(result)
        df["platform"] = app_name
        all_reviews.append(df)
        print(f"✅ Collected {len(df)} reviews for {app_name}")
    except Exception as e:
        print(f"❌ Failed for {app_name}: {e}")

if all_reviews:
    final_df = pd.concat(all_reviews, ignore_index=True)
    final_df.to_csv("../../data/raw/google_play_ecommerce_reviews_limited.csv", index=False, encoding="utf-8")
    print("✅ Saved as google_play_ecommerce_reviews_limited.csv")
    print(final_df["platform"].value_counts())
else:
    print("⚠️ No reviews collected. Please check package IDs and internet connection.")
