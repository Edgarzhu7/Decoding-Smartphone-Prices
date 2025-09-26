import json
import os
from datetime import datetime, timedelta

def parse_keepa_csv(csv_data):
    base_date = datetime(2011, 1, 1)
    readable_prices = []

    for i in range(0, len(csv_data), 2):
        timestamp = csv_data[i]
        price_cents = csv_data[i + 1]

        if price_cents == -1:
            continue

        date = base_date + timedelta(minutes=timestamp)
        price = round(price_cents / 100.0, 2)

        readable_prices.append({
            "date": date.isoformat(),
            "price": price
        })

    return readable_prices

def convert_price_history(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    if "csv" not in data or len(data["csv"]) == 0 or not data["csv"][1]:
        print(f"Skipping {input_file}: no price history.")
        return

    csv_price_data = data["csv"][1]  # Amazon NEW price
    asin = data.get("asin", "Unknown")
    readable_history = parse_keepa_csv(csv_price_data)

    with open(output_file, "w") as f:
        json.dump({"price_history": readable_history, "asin": asin}, f, indent=4)

    print(f"Saved readable price history to {output_file}")

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            convert_price_history(input_path, output_path)

if __name__ == "__main__":
    input_folder = "product_data_single"           # input folder with csv-style JSON
    output_folder = "readable_price_history_single"  # output folder with readable price history
    process_folder(input_folder, output_folder)
