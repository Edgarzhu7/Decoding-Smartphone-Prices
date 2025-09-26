import json
from datetime import datetime


def process_amazon_price_history(product_json_path, output_history_path=None):
    with open(product_json_path, 'r') as f:
        product_data = json.load(f)

    amazon_price_history = product_data['csv'][0]

    if not amazon_price_history:
        print(f"No Amazon price history available for {product_json_path}")
        return []

    processed_history = []
    for i in range(0, len(amazon_price_history), 2):
        keepa_time = amazon_price_history[i]
        price_cents = amazon_price_history[i + 1]

        timestamp = datetime.utcfromtimestamp((keepa_time + 21564000) * 60)
        price = price_cents / 100 if price_cents != -1 else None

        processed_history.append({"date": timestamp.isoformat(), "price": price})

    if output_history_path:
        with open(output_history_path, 'w') as f:
            json.dump(processed_history, f, indent=4)

        print(f"Price history saved to {output_history_path}")

    return processed_history


if __name__ == "__main__":
    product_json_path = "product_data/B09G99QJSK.json"
    output_history_path = "product_data/B09G99QJSK_price_history.json"

    history = process_amazon_price_history(product_json_path, output_history_path)

    for entry in history[:10]:
        print(entry)
