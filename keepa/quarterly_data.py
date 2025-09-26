# import json
# import sys
# from collections import defaultdict
# from datetime import datetime

# def compute_quarterly_averages(input_filename, output_filename):
#     # Read JSON input from file
#     with open(input_filename, "r") as f:
#         parsed_data = json.load(f)
    
#     # Dictionary to store prices by year and quarter
#     price_by_quarter = defaultdict(lambda: defaultdict(list))
    
#     # Process price history
#     for entry in parsed_data["price_history"]:
#         date_str = entry["date"]
#         price = entry["price"]
        
#         if price is not None:  # Ignore None prices
#             date_obj = datetime.fromisoformat(date_str)
#             year = date_obj.year
#             quarter = (date_obj.month - 1) // 3 + 1
            
#             price_by_quarter[year][quarter].append(price)
    
#     # Compute average price per quarter
#     avg_price_by_quarter = {}
#     for year, quarters in price_by_quarter.items():
#         avg_price_by_quarter[year] = {}
#         for quarter, prices in quarters.items():
#             avg_price_by_quarter[year][quarter] = sum(prices) / len(prices)
    
#     # Save results to a file
#     with open(output_filename, "w") as f:
#         json.dump(avg_price_by_quarter, f, indent=4)
    
#     print(f"Quarterly average prices saved to {output_filename}")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python script.py <input_json_file> <output_json_file>")
#         sys.exit(1)
    
#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
    
#     compute_quarterly_averages(input_file, output_file)
import json
import sys
import os
from collections import defaultdict
from datetime import datetime

def compute_quarterly_averages(input_filename):
    # Read JSON input from file
    with open(input_filename, "r") as f:
        parsed_data = json.load(f)
    
    # Dictionary to store prices by year and quarter
    price_by_quarter = defaultdict(lambda: defaultdict(list))
    
    asin = parsed_data.get("asin", "Unknown")
    # Process price history
    for entry in parsed_data.get("price_history", []):
        date_str = entry["date"]
        price = entry["price"]
        
        if price is not None:  # Ignore None prices
            date_obj = datetime.fromisoformat(date_str)
            year = date_obj.year
            quarter = (date_obj.month - 1) // 3 + 1
            
            price_by_quarter[year][quarter].append(price)
    
    # Compute average price per quarter
    avg_price_by_quarter = {}
    for year, quarters in price_by_quarter.items():
        avg_price_by_quarter[year] = {}
        for quarter, prices in quarters.items():
            avg_price_by_quarter[year][quarter] = sum(prices) / len(prices)
    
    # Fill missing quarters with the average of surrounding quarters
    years = sorted(avg_price_by_quarter.keys())
    for year in years:
        for quarter in range(1, 5):
            if quarter not in avg_price_by_quarter[year]:
                prev_quarter = avg_price_by_quarter[year].get(quarter - 1)
                next_quarter = avg_price_by_quarter[year].get(quarter + 1)
                
                # If previous and next quarters exist within the same year, use their average
                if prev_quarter is not None and next_quarter is not None:
                    avg_price_by_quarter[year][quarter] = (prev_quarter + next_quarter) / 2
    
    # Special case: If Q4 is missing in a year but both Q3 of the same year and Q1 of the next year exist, use their average
    for i in range(len(years) - 1):
        current_year = years[i]
        next_year = years[i + 1]
        
        if 4 not in avg_price_by_quarter[current_year]:
            q3 = avg_price_by_quarter[current_year].get(3)
            q1_next = avg_price_by_quarter[next_year].get(1)
            
            if q3 is not None and q1_next is not None:
                avg_price_by_quarter[current_year][4] = (q3 + q1_next) / 2
    
    return avg_price_by_quarter if avg_price_by_quarter else None, asin

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            avg_price_by_quarter, asin = compute_quarterly_averages(input_filepath)
            
            if avg_price_by_quarter:  # Only save if there is valid price data
                with open(output_filepath, "w") as f:
                    json.dump({"price_history": avg_price_by_quarter, "asin": asin}, f, indent=4)
                print(f"Processed {filename} and saved results to {output_filepath}")
            else:
                print(f"Skipping {filename} as it contains no valid price data")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    process_folder(input_folder, output_folder)
