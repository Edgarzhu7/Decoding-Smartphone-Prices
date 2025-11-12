import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import from Quarter folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Quarter'))
from lasso_price_prediction import preprocess_features

def aggregate_quarters_to_years(df):
    """
    Aggregate quarterly price data to annual averages
    Returns a new DataFrame with annual columns
    """
    # Get all quarter columns
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    
    # Group quarters by year
    year_data = {}
    for col in quarter_columns:
        try:
            parts = col.split()
            if len(parts) >= 2 and 'Q' in parts[1]:
                year = int(parts[0])
                if year not in year_data:
                    year_data[year] = []
                year_data[year].append(col)
        except:
            continue
    
    # Create annual DataFrame with non-price columns
    annual_df = df[['Company Name', 'Model Name', 'ASIN', 'Mobile Weight', 'Ram Mem', 
                    'Front Camera', 'Back Camera', 'Max_MP', 'Num_Cameras', 
                    'Processor', 'Processor Level', 'Battery Capacity', 'Screen Size', 'Resolution']].copy()
    
    # Calculate annual average prices
    for year in sorted(year_data.keys()):
        year_cols = year_data[year]
        # Calculate mean of available quarters for each product
        annual_prices = df[year_cols].mean(axis=1)
        annual_df[str(year)] = annual_prices
    
    return annual_df

def parse_year_column(col_name):
    """
    Parse year column name, return year
    Example: "2019" -> 2019
    """
    try:
        year = int(col_name)
        return year
    except:
        return None

def calculate_annual_jevons_index(df, year1_col, year2_col):
    """
    Calculate Jevons index between two years
    
    Parameters:
    df: DataFrame containing annual price data
    year1_col: Base period year column name (t-1)
    year2_col: Current period year column name (t)
    
    Returns:
    Jevons index value, number of valid products
    """
    
    if year1_col not in df.columns or year2_col not in df.columns:
        return None, 0
    
    # Get price data for both years
    year1_prices = df[year1_col]
    year2_prices = df[year2_col]
    
    # Filter out missing values or zero values
    valid_mask = (year1_prices > 0) & (year2_prices > 0) & \
                 (~year1_prices.isna()) & (~year2_prices.isna())
    
    if valid_mask.sum() == 0:
        return None, 0
    
    y1_valid = year1_prices[valid_mask]
    y2_valid = year2_prices[valid_mask]
    
    # Calculate log price ratios
    log_price_ratios = np.log(y2_valid) - np.log(y1_valid)
    
    # Calculate Jevons index (mean log difference, without exp)
    jevons_index = float(np.mean(log_price_ratios))
    
    return jevons_index, len(y1_valid)

def get_sorted_year_columns(df):
    """
    Get sorted year columns
    """
    year_columns = []
    for col in df.columns:
        year = parse_year_column(col)
        if year is not None:
            year_columns.append(col)
    
    year_columns = sorted(year_columns, key=lambda x: int(x))
    return year_columns

def calculate_adjacent_annual_indices(df):
    """
    Calculate Jevons indices between adjacent years
    """
    year_columns = get_sorted_year_columns(df)
    
    results = []
    
    # Calculate indices between adjacent years
    for i in range(len(year_columns) - 1):
        year1_col = year_columns[i]
        year2_col = year_columns[i + 1]
        
        jevons_index, n_products = calculate_annual_jevons_index(df, year1_col, year2_col)
        
        if jevons_index is not None:
            results.append({
                'Base Year': year1_col,
                'Current Year': year2_col,
                'Period': f"{year1_col} → {year2_col}",
                'Jevons Index': jevons_index,
                'Number of Products': n_products,
                'Price Change (%)': jevons_index * 100
            })
            
            print(f"Jevons Index ({year1_col} → {year2_col}): {jevons_index:.6f} ({n_products} products)")
    
    return pd.DataFrame(results)

def calculate_all_annual_pairs(df):
    """
    Calculate Jevons indices for all year pairs
    """
    year_columns = get_sorted_year_columns(df)
    
    results = []
    
    # Calculate indices for all year pairs
    for i in range(len(year_columns)):
        for j in range(i + 1, len(year_columns)):
            year1_col = year_columns[i]
            year2_col = year_columns[j]
            
            jevons_index, n_products = calculate_annual_jevons_index(df, year1_col, year2_col)
            
            if jevons_index is not None:
                year1 = int(year1_col)
                year2 = int(year2_col)
                years_apart = year2 - year1
                
                results.append({
                    'Base Year': year1_col,
                    'Current Year': year2_col,
                    'Period': f"{year1_col} → {year2_col}",
                    'Jevons Index': jevons_index,
                    'Number of Products': n_products,
                    'Price Change (%)': jevons_index * 100,
                    'Years Apart': years_apart
                })
    
    return pd.DataFrame(results)

def main():
    """
    Main function: Read data, aggregate to annual, calculate annual Jevons indices, and output to Excel
    """
    print("Reading Dataset.xlsx...")
    df = pd.read_excel('../Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Aggregate quarterly data to annual data
    print("\nAggregating quarterly data to annual averages...")
    annual_df = aggregate_quarters_to_years(df)
    
    # Get all year columns
    year_columns = get_sorted_year_columns(annual_df)
    print(f"Annual data columns: {year_columns}")
    
    # Calculate adjacent year Jevons indices
    print("\n=== Calculating Adjacent Year Jevons Indices ===")
    adjacent_results = calculate_adjacent_annual_indices(annual_df)
    
    # Calculate all year pair Jevons indices
    print(f"\n=== Calculating All Year Pair Jevons Indices (Total {len(year_columns)*(len(year_columns)-1)//2} pairs) ===")
    all_pairs_results = calculate_all_annual_pairs(annual_df)
    print(f"Actually calculated {len(all_pairs_results)} valid year pairs")
    
    # Output to Excel file
    output_file = 'Annual_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Adjacent year results
        adjacent_results.to_excel(writer, sheet_name='Adjacent Years', index=False)
        
        # All year pair results
        all_pairs_results.to_excel(writer, sheet_name='All Year Pairs', index=False)
        
        # Annual price data
        annual_df.to_excel(writer, sheet_name='Annual_Price_Data', index=False)
        
        # Data summary
        summary_data = {
            'Metric': [
                'Total Products', 
                'Total Years', 
                'Adjacent Year Comparisons', 
                'Total Year Pair Comparisons'
            ],
            'Value': [
                len(annual_df), 
                len(year_columns),
                len(adjacent_results),
                len(all_pairs_results)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Adjacent year comparisons: {len(adjacent_results)}")
    print(f"All year pair comparisons: {len(all_pairs_results)}")
    
    # Display some key results
    if not adjacent_results.empty:
        print("\n=== Adjacent Year Jevons Index Summary ===")
        display_df = adjacent_results[['Period', 'Jevons Index', 'Price Change (%)', 'Number of Products']]
        print(display_df.to_string(index=False))
    
    return adjacent_results, all_pairs_results, annual_df

if __name__ == "__main__":
    adjacent_results, all_pairs_results, annual_df = main()

