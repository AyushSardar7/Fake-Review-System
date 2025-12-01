import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.flipkart.com/apple-iphone-14-plus-blue-128-gb/product-reviews/itm04b14e7ff6b90?pid=MOBGHWFHQFSQYBFU&page=1"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# New class for reviews
reviews = soup.find_all("div", {"class": "ZmyHeo"})

review_list = []
for r in reviews:
    review_list.append(r.text.strip())
    print(r.text.strip())   # print reviews in terminal

# Save to CSV
df = pd.DataFrame(review_list, columns=["review_text"])
df.to_csv("../../data/raw/flipkart_reviews.csv", index=False)
print("✅ Reviews saved to flipkart_reviews.csv")
