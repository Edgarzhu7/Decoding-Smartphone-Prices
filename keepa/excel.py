import json
import os
import csv

def extract_flat_row(json_data):
    asin = json_data.get("asin", "UNKNOWN")
    history = json_data.get("price_history", {})
    flat_row = {"asin": asin}

    for year in history:
        for quarter in history[year]:
            key = f"{year} Q{quarter}"
            flat_row[key] = round(history[year][quarter], 2)

    return flat_row

def batch_convert_to_csv(input_folder, output_csv):
    all_rows = []
    all_keys = set()

    # Read and flatten all files
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            with open(os.path.join(input_folder, filename), "r") as f:
                data = json.load(f)
                flat_row = extract_flat_row(data)
                all_keys.update(flat_row.keys())
                all_rows.append(flat_row)

    # Ensure consistent column order: asin first, then sorted quarter keys
    all_keys = sorted(k for k in all_keys if k != "asin")
    fieldnames = ["asin"] + all_keys

    # Sort rows by asin
    all_rows.sort(key=lambda row: row["asin"])

    # Write to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Saved combined CSV to: {output_csv}")

if __name__ == "__main__":
    input_folder = "quarterly_price_single"
    output_csv = "combined_price_history_single.csv"
    batch_convert_to_csv(input_folder, output_csv)
