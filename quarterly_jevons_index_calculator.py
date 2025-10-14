import pandas as pd
import numpy as np
import math
from itertools import combinations

def parse_quarter_column(col_name):
    """
    Parse quarter column name, return (year, quarter)
    Example: "2019 Q1" -> (2019, 1)
    """
    try:
        parts = col_name.split()
        if len(parts) >= 2 and 'Q' in parts[1]:
            year = int(parts[0])
            quarter = int(parts[1].replace('Q', ''))
            return year, quarter
    except:
        pass
    return None, None

def get_quarter_order(year, quarter):
    """
    Get global order number for quarters
    Example: 2019 Q1 -> 1, 2019 Q2 -> 2, ..., 2020 Q1 -> 5
    """
    base_year = 2018
    base_quarter = 4  # Starting from 2018 Q4
    
    # Calculate quarters from base point
    total_quarters = (year - base_year) * 4 + quarter - base_quarter
    return total_quarters

def calculate_quarterly_jevons_index(df, quarter1_col, quarter2_col):
    """
    Calculate Jevons index between two quarters
    
    Parameters:
    df: DataFrame containing price data
    quarter1_col: Base period quarter column name (t-1)
    quarter2_col: Current period quarter column name (t)
    
    Returns:
    Jevons index value, number of valid products
    """
    
    if quarter1_col not in df.columns or quarter2_col not in df.columns:
        return None, 0
    
    # Get price data for both quarters
    quarter1_prices = df[quarter1_col]
    quarter2_prices = df[quarter2_col]
    
    # Filter out missing values or zero values
    valid_mask = (quarter1_prices > 0) & (quarter2_prices > 0) & \
                 (~quarter1_prices.isna()) & (~quarter2_prices.isna())
    
    if valid_mask.sum() == 0:
        return None, 0
    
    q1_valid = quarter1_prices[valid_mask]
    q2_valid = quarter2_prices[valid_mask]
    
    # Calculate log price ratios
    log_price_ratios = np.log(q2_valid) - np.log(q1_valid)
    
    # Calculate Jevons index (geometric mean)
    # I^Jevons = exp((1/N) * Σ(ln P_{i,t} - ln P_{i,t-1}))
    jevons_index = np.exp(np.mean(log_price_ratios))
    
    return jevons_index, len(q1_valid)

def calculate_adjacent_quarterly_indices(df):
    """
    Calculate Jevons indices between adjacent quarters
    """
    # Get all quarter columns and sort them
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    
    # Sort by chronological order
    quarter_data = []
    for col in quarter_columns:
        year, quarter = parse_quarter_column(col)
        if year is not None and quarter is not None:
            order = get_quarter_order(year, quarter)
            quarter_data.append((order, col, year, quarter))
    
    quarter_data.sort(key=lambda x: x[0])  # Sort by time order
    sorted_columns = [item[1] for item in quarter_data]
    
    results = []
    
    # Calculate indices between adjacent quarters
    for i in range(len(sorted_columns) - 1):
        quarter1_col = sorted_columns[i]
        quarter2_col = sorted_columns[i + 1]
        
        jevons_index, n_products = calculate_quarterly_jevons_index(df, quarter1_col, quarter2_col)
        
        if jevons_index is not None:
            results.append({
                'Base Quarter': quarter1_col,
                'Current Quarter': quarter2_col,
                'Period': f"{quarter1_col} → {quarter2_col}",
                'Jevons Index': jevons_index,
                'Number of Products': n_products,
                'Price Change (%)': (jevons_index - 1) * 100
            })
            
            print(f"Jevons Index ({quarter1_col} → {quarter2_col}): {jevons_index:.4f} ({n_products} products)")
    
    return pd.DataFrame(results)

def calculate_all_quarterly_pairs(df):
    """
    Calculate Jevons indices for all quarter pairs
    """
    # Get all quarter columns and sort them
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    
    # Sort by chronological order
    quarter_data = []
    for col in quarter_columns:
        year, quarter = parse_quarter_column(col)
        if year is not None and quarter is not None:
            order = get_quarter_order(year, quarter)
            quarter_data.append((order, col, year, quarter))
    
    quarter_data.sort(key=lambda x: x[0])
    sorted_columns = [item[1] for item in quarter_data]
    
    results = []
    
    # Calculate indices for all quarter pairs
    for i in range(len(sorted_columns)):
        for j in range(i + 1, len(sorted_columns)):
            quarter1_col = sorted_columns[i]
            quarter2_col = sorted_columns[j]
            
            jevons_index, n_products = calculate_quarterly_jevons_index(df, quarter1_col, quarter2_col)
            
            if jevons_index is not None:
                # Calculate quarter interval
                year1, q1 = parse_quarter_column(quarter1_col)
                year2, q2 = parse_quarter_column(quarter2_col)
                quarters_apart = (year2 - year1) * 4 + (q2 - q1)
                
                results.append({
                    'Base Quarter': quarter1_col,
                    'Current Quarter': quarter2_col,
                    'Period': f"{quarter1_col} → {quarter2_col}",
                    'Jevons Index': jevons_index,
                    'Number of Products': n_products,
                    'Price Change (%)': (jevons_index - 1) * 100,
                    'Quarters Apart': quarters_apart
                })
    
    return pd.DataFrame(results)

def calculate_same_quarter_across_years(df):
    """
    Calculate Jevons indices for same quarter across years (e.g., 2019 Q1 vs 2020 Q1)
    """
    # Group by quarter type
    quarters_by_type = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    
    for col in df.columns:
        if 'Q' in col:
            year, quarter = parse_quarter_column(col)
            if year is not None and quarter is not None:
                quarter_type = f'Q{quarter}'
                if quarter_type in quarters_by_type:
                    quarters_by_type[quarter_type].append((year, col))
    
    # Sort each quarter type by year
    for quarter_type in quarters_by_type:
        quarters_by_type[quarter_type].sort(key=lambda x: x[0])
    
    results = []
    
    # Calculate indices for same quarter across years
    for quarter_type, quarter_list in quarters_by_type.items():
        if len(quarter_list) < 2:
            continue
            
        for i in range(len(quarter_list) - 1):
            year1, col1 = quarter_list[i]
            year2, col2 = quarter_list[i + 1]
            
            jevons_index, n_products = calculate_quarterly_jevons_index(df, col1, col2)
            
            if jevons_index is not None:
                results.append({
                    'Quarter Type': quarter_type,
                    'Base Year': year1,
                    'Current Year': year2,
                    'Base Quarter': col1,
                    'Current Quarter': col2,
                    'Period': f"{col1} → {col2}",
                    'Jevons Index': jevons_index,
                    'Number of Products': n_products,
                    'Price Change (%)': (jevons_index - 1) * 100,
                    'Years Apart': year2 - year1
                })
    
    return pd.DataFrame(results)

def main():
    """
    Main function: Read data, calculate quarterly Jevons indices, and output to Excel
    """
    print("Reading Dataset.xlsx...")
    df = pd.read_excel('Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Get all quarter columns
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    print(f"Quarter data columns: {quarter_columns}")
    
    # Calculate adjacent quarter Jevons indices
    print("\n=== Calculating Adjacent Quarter Jevons Indices ===")
    adjacent_results = calculate_adjacent_quarterly_indices(df)
    
    # Calculate all quarter pair Jevons indices
    print(f"\n=== Calculating All Quarter Pair Jevons Indices (Total {len(quarter_columns)*(len(quarter_columns)-1)//2} pairs) ===")
    all_pairs_results = calculate_all_quarterly_pairs(df)
    print(f"Actually calculated {len(all_pairs_results)} valid quarter pairs")
    
    # Calculate same quarter across years indices
    print("\n=== Calculating Same Quarter Across Years Jevons Indices ===")
    same_quarter_results = calculate_same_quarter_across_years(df)
    
    # Output to Excel file
    output_file = 'Quarterly_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Adjacent quarter results
        adjacent_results.to_excel(writer, sheet_name='Adjacent Quarters', index=False)
        
        # All quarter pair results
        all_pairs_results.to_excel(writer, sheet_name='All Quarter Pairs', index=False)
        
        # Same quarter across years results
        same_quarter_results.to_excel(writer, sheet_name='Same Quarter Across Years', index=False)
        
        # Data summary
        summary_data = {
            'Metric': [
                'Total Products', 
                'Total Quarters', 
                'Adjacent Quarter Comparisons', 
                'Total Quarter Pair Comparisons',
                'Same Quarter Across Years Comparisons'
            ],
            'Value': [
                len(df), 
                len(quarter_columns),
                len(adjacent_results),
                len(all_pairs_results),
                len(same_quarter_results)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Adjacent quarter comparisons: {len(adjacent_results)}")
    print(f"All quarter pair comparisons: {len(all_pairs_results)}")
    print(f"Same quarter across years comparisons: {len(same_quarter_results)}")
    
    # Display some key results
    if not adjacent_results.empty:
        print("\n=== Adjacent Quarter Jevons Index Summary (First 10) ===")
        display_df = adjacent_results[['Period', 'Jevons Index', 'Price Change (%)', 'Number of Products']].head(10)
        print(display_df.to_string(index=False))
    
    if not same_quarter_results.empty:
        print("\n=== Same Quarter Across Years Jevons Index Summary (First 10) ===")
        display_df = same_quarter_results[['Quarter Type', 'Period', 'Jevons Index', 'Price Change (%)', 'Number of Products']].head(10)
        print(display_df.to_string(index=False))
    
    return adjacent_results, all_pairs_results, same_quarter_results

if __name__ == "__main__":
    adjacent_results, all_pairs_results, same_quarter_results = main()
