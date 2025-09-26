import requests
import json
import time
import os
import re
from datetime import datetime
import urllib.parse



API_KEY = "c6ffh6j8p305l28qa6alk6ojvhjeadlsrdf2litlrr50pc1vs192vurvtsnrr908"
DOMAIN_ID = 1  # Amazon.com (US)

# Search terms for major phone brands
# SEARCH_TERMS = ["iphone"] #98
# SEARCH_TERMS = ["Samsung%20galaxy%20M"] #80
# SEARCH_TERMS = ["Samsung%20galaxy%20Note"] #80
# SEARCH_TERMS = ["Oneplus%20phone"] #60
# SEARCH_TERMS = ["Vivo%20phone"] #80

# SEARCH_TERMS = ["Oppo%20phone"] #80
SEARCH_TERMS = ["Realme%20phone"] #80
# SEARCH_TERMS = ["Xiaomi%20xiaomi", "Redmi"] #40
# SEARCH_TERMS = ["Lenovo%20phone"] #20
# SEARCH_TERMS = ["Motorola%20moto", "Motorola%20edge"] #60
# SEARCH_TERMS = ["Huawei%20phone"] #40
# SEARCH_TERMS = ["Nokia%20G"] #10
# SEARCH_TERMS = ["Sony%20Xperia"] #10
# SEARCH_TERMS = ["Google%20pixel"] #20
# SEARCH_TERMS = ["Tecno%20phone"] #20
# SEARCH_TERMS = ["Infinix%20hot"] #30
# SEARCH_TERMS = ["Honor%20phone"] #60
# SEARCH_TERMS = ["POCO%20phone"] #30




#, "samsung%20phone", "google%20phone", "oneplus%20phone", "motorola%20phone", "xiaomi%20phone", "sony%20phone"
# SEARCH_TERMS = [urllib.parse.quote_plus(term) for term in [
#     "apple phone"
# ]]
    # , "samsung phone", "google phone", 
    # "oneplus phone", "motorola phone", "xiaomi phone", "sony phone"
OUTPUT_DIR = "phone_price_history1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def search_products(search_term, max_results=60):
    results = []
    BASE_URL = "https://api.keepa.com/search"
    # page = 0

    # while len(results) < max_results:  # Each page gives 10 results; get up to 3 pages
    # params = {
    #     "key": "API_KEY",
    #     "domain": DOMAIN_ID,
    #     "type": "product",
    #     "term": search_term,
    #     "history": 1,  # Get price history
    #     "page": 0,
    #     # "stats": "2011-01-01,2025-01-01",  # Get entire history for stats
    #     # "page": page,
    #     # "update": 0  # Triggers a fresh search sometimes
    # }
    # /search?key=<yourAccessKey>&domain=<domainId>&type=product&term=<searchTerm>
    # https://api.keepa.com/search?key=c6ffh6j8p305l28qa6alk6ojvhjeadlsrdf2litlrr50pc1vs192vurvtsnrr908&domain=1&type=product&term=apple&history=1
    for i in range(4):
        url = f"{BASE_URL}?key={API_KEY}&domain={DOMAIN_ID}&type=product&term={search_term}&history=1&page={i}"
        # response = requests.get("https://api.keepa.com/search", params=params)
        response = requests.get(url)
        data = response.json()

        if "products" in data:
            results.extend(data["products"])
        else:
            print(f"No products found for {search_term}")
                # break

            # page += 1
            # time.sleep(2)  # Respect API rate limits

    return results[:max_results]


def process_price_history(product):
    asin = product.get("asin")
    title = product.get("title", "Unknown Product")

    amazon_price_history = product.get("csv", [None])[0]
    if not amazon_price_history:
        print(f"No Amazon price history available for {title} ({asin})")
        return None

    processed_price_history = []
    for i in range(0, len(amazon_price_history), 2):
        keepa_time = amazon_price_history[i]
        price_cents = amazon_price_history[i + 1]

        timestamp = datetime.utcfromtimestamp((keepa_time + 21564000) * 60)
        price = price_cents / 100 if price_cents != -1 else None

        processed_price_history.append({"date": timestamp.isoformat(), "price": price})

    # Extract Sales Rank history (if available)
    sales_ranks = product.get("salesRanks", {})
    processed_sales_rank_history = {}

    for category_id, sales_data in sales_ranks.items():
        category_history = []
        for i in range(0, len(sales_data), 2):
            keepa_time = sales_data[i]
            rank = sales_data[i + 1]

            timestamp = datetime.utcfromtimestamp((keepa_time + 21564000) * 60)

            category_history.append({"date": timestamp.isoformat(), "sales_rank": rank})

        processed_sales_rank_history[category_id] = category_history

    return {
        "asin": asin,
        "title": title,
        "price_history": processed_price_history,
        "sales_rank_history": processed_sales_rank_history
    }


def main():
    for term in SEARCH_TERMS:
        print(f"Searching for {term}...")
        products = search_products(term)

        for product in products:
            asin = product.get("asin")
            title = product.get("title", "Unknown Product")
            if not asin:
                continue

            price_data = process_price_history(product)
            if price_data:
                safe_filename = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(" ", "_"))
                output_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.json")
                # output_path = os.path.join(OUTPUT_DIR, f"{title}.json")
                with open(output_path, 'w') as f:
                    json.dump(price_data, f, indent=4)

                print(f"Saved price history for {price_data['title']} ({asin}) to {output_path}")


if __name__ == "__main__":
    main()
