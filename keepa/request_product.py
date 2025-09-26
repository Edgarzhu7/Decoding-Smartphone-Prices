# import requests
# import json
# import os

# def request_keepa_product(api_key, asin, domain=1, output_dir="product_data"):
#     url = "https://api.keepa.com/product"
#     params = {
#         "key": api_key,
#         "domain": domain,  # 1 for Amazon.com (US)
#         "asin": asin,
#         "stats": "2011-01-01,2025-01-01",  # Entire history for stats
#         "history": 1,  # Ensure we get price history data
#         "code-limit": 1,  # No limit on the number of codes
#     }

#     response = requests.get(url, params=params)

#     if response.status_code == 200:
#         data = response.json()
#         if 'products' in data and len(data['products']) > 0:
#             product = data['products'][0]

#             # Ensure output directory exists
#             os.makedirs(output_dir, exist_ok=True)

#             title = product.get("title", "Unknown Product")

#             # Save product data to a file
#             output_path = os.path.join(output_dir, f"{title}.json")
#             with open(output_path, 'w') as f:
#                 json.dump(product, f, indent=4)
            
            

#             print(f"Product data saved to {title}")
#         else:
#             print(f"No product data found for ASIN: {asin}")
#     else:
#         print(f"Failed to retrieve data: {response.status_code}, {response.text}")

# if __name__ == "__main__":
#     API_KEY = "c6ffh6j8p305l28qa6alk6ojvhjeadlsrdf2litlrr50pc1vs192vurvtsnrr908"
#     ASIN = "B0DHJGKNT1"  # Example ASIN for Samsung Galaxy A14 5G
#     request_keepa_product(API_KEY, ASIN)
import requests
import json
import re
import os

def sanitize_filename(name):
    """Remove unsafe characters from filename."""
    return re.sub(r'[\\\\/:*?"<>|]', '', name).strip()[:100]  # Remove bad chars & limit length


def request_keepa_product(api_key, asin, domain=1, output_dir="product_data_single"):
    url = "https://api.keepa.com/product"
    params = {
        "key": api_key,
        "domain": domain,
        "asin": asin,
        "stats": "2011-01-01,2025-01-01",
        "history": 1,
        "code-limit": 1,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'products' in data and len(data['products']) > 0:
            product = data['products'][0]

            os.makedirs(output_dir, exist_ok=True)

            title = product.get("title", "Unknown Product")
            if title:
                filename = sanitize_filename(title) + ".json"
            else:
                filename = asin + ".json"
            # Use ASIN instead of title to avoid unsafe characters in filenames
            output_path = os.path.join(output_dir, f"{filename}.json")
            with open(output_path, 'w') as f:
                json.dump(product, f, indent=4)

            print(f"Product data saved to {output_path}")
        else:
            print(f"No product data found for ASIN: {asin}")
    else:
        print(f"Failed to retrieve data for ASIN {asin}: {response.status_code}, {response.text}")

if __name__ == "__main__":
    API_KEY = "c6ffh6j8p305l28qa6alk6ojvhjeadlsrdf2litlrr50pc1vs192vurvtsnrr908"

    asin_file = "Asincode"
    if not os.path.isfile(asin_file):
        print(f"ASIN file '{asin_file}' not found.")
    else:
        with open(asin_file, "r") as file:
            asin_list = [line.strip() for line in file if line.strip()]
        
        for asin in asin_list:
            request_keepa_product(API_KEY, asin)
