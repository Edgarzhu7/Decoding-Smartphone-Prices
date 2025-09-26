import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def plot_price_history(json_file, output_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    product_title = data.get("title", "Unknown Product")
    price_data = data.get("price_history", [])
    
    if not price_data:
        print(f"No price history available for {product_title}")
        return
    
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df['price'].ffill()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Price History for {product_title}')
    plt.xticks(rotation=45)
    plt.grid()
    
    # Sanitize filename
    safe_filename = product_title.replace(" ", "_").replace("/", "-").replace("?", "")
    output_path = os.path.join(output_folder, f"{safe_filename}.png")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved price history plot for {product_title} to {output_path}")

def plot_all_price_histories(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_file = os.path.join(input_folder, filename)
            plot_price_history(json_file, output_folder)

# Example usage
# plot_all_price_histories('phone_price_history', 'plots')
if __name__ == "__main__":
    plot_all_price_histories('phone_price_history', 'plots')