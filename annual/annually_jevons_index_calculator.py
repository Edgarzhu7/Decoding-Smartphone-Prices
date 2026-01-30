import pandas as pd
import numpy as np
import sys
import os
from typing import List

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
    # Get all non-quarter columns (feature columns)
    feature_cols_list = ['Company Name', 'Model Name', 'Mobile Weight', 'RAM', 
                        'Front Camera', 'Back Camera', 'Max_MP', 'Num_Cameras', 
                        'Processor', 'Processor Level', 'Battery Capacity', 'Screen Size']
    # Add ASIN columns if they exist
    asin_cols = [col for col in df.columns if 'ASIN' in col]
    feature_cols_list = [col for col in feature_cols_list if col in df.columns] + asin_cols
    
    annual_df = df[feature_cols_list].copy()
    
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

def create_model_period_matrix(annual_df: pd.DataFrame, years: List[str]):
    """
    Create a matrix with Model Name (or Company Name + Model Name) as columns and year pairs as rows
    Each cell contains log_delta (log price change) for that model in that period
    Only fills values for periods where the model has prices in both years
    
    Args:
        annual_df: DataFrame with Company Name, Model Name, and year price columns
        years: List of all year column names (sorted, as strings)
    
    Returns:
        DataFrame with year pairs as index and Model identifiers as columns
    """
    if annual_df.empty:
        print('Warning: annual_df is empty, cannot create matrix')
        return pd.DataFrame()
    
    # Check required columns
    if 'Model Name' not in annual_df.columns:
        print('Error: Model Name column not found in annual_df')
        return pd.DataFrame()
    
    # Create model identifier: use Model Name if available, otherwise Company Name + Model Name
    df_copy = annual_df.copy()
    if 'Company Name' in df_copy.columns:
        df_copy['Model_Identifier'] = df_copy['Company Name'].astype(str) + ' - ' + df_copy['Model Name'].astype(str)
    else:
        df_copy['Model_Identifier'] = df_copy['Model Name'].astype(str)
    
    # Get all unique year pairs (sorted)
    year_pairs = []
    for i in range(len(years) - 1):
        year_prev, year_curr = years[i], years[i + 1]
        year_pairs.append(f"{year_prev}→{year_curr}")
    
    # Get all unique model identifiers, preserving original dataset order
    # Use drop_duplicates() instead of unique() to preserve first occurrence order
    model_identifiers = df_copy['Model_Identifier'].dropna().drop_duplicates().tolist()
    
    if len(model_identifiers) == 0:
        print('Warning: No model identifiers found in annual_df')
        return pd.DataFrame()
    
    print(f'Found {len(model_identifiers)} models and {len(year_pairs)} year pairs')
    
    # Create empty matrix with all year pairs and all models
    matrix_data = {}
    for model_id in model_identifiers:
        matrix_data[model_id] = [np.nan] * len(year_pairs)
    
    # Fill matrix with log_delta values
    filled_count = 0
    for _, row in df_copy.iterrows():
        model_id = row['Model_Identifier']
        
        # For each year pair, calculate log delta if both prices exist
        for i in range(len(years) - 1):
            year_prev, year_curr = years[i], years[i + 1]
            
            # Check if both prices exist and are valid
            if (year_prev in row.index and year_curr in row.index and
                pd.notna(row[year_prev]) and pd.notna(row[year_curr]) and
                row[year_prev] > 0 and row[year_curr] > 0):
                
                # Calculate log delta
                log_delta = np.log(row[year_curr]) - np.log(row[year_prev])
                
                year_str = f"{year_prev}→{year_curr}"
                if year_str in year_pairs:
                    year_idx = year_pairs.index(year_str)
                    if model_id in matrix_data:
                        # If multiple entries for same model-period (multiple ASINs), take mean
                        if pd.isna(matrix_data[model_id][year_idx]):
                            matrix_data[model_id][year_idx] = log_delta
                        else:
                            # Average if multiple ASINs for same model
                            matrix_data[model_id][year_idx] = (matrix_data[model_id][year_idx] + log_delta) / 2
                        filled_count += 1
    
    print(f'Filled {filled_count} cells in the matrix')
    
    # Create DataFrame with year pairs as index
    matrix_df = pd.DataFrame(matrix_data, index=year_pairs)
    
    return matrix_df

def main():
    """
    Main function: Read data, aggregate to annual, calculate annual Jevons indices, and output to Excel
    """
    print("Reading Dataset.xlsx...")
    # Get path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'Dataset.xlsx')
    df = pd.read_excel(dataset_path)
    
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

    # Create model-period matrix
    print("\n=== Creating Model-Period Matrix ===")
    try:
        model_period_matrix = create_model_period_matrix(annual_df, year_columns)
        if model_period_matrix.empty:
            print('Warning: Model-period matrix is empty!')
        else:
            print(f'Model-period matrix created: {model_period_matrix.shape[0]} periods × {model_period_matrix.shape[1]} models')
    except Exception as e:
        print(f'Error creating model-period matrix: {e}')
        import traceback
        traceback.print_exc()
        model_period_matrix = pd.DataFrame()
    
    # Output to Excel file
    output_file = 'Annual_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Adjacent year results
        adjacent_results.to_excel(writer, sheet_name='Adjacent Years', index=False)
        
        # All year pair results
        all_pairs_results.to_excel(writer, sheet_name='All Year Pairs', index=False)
        
        # Annual price data
        annual_df.to_excel(writer, sheet_name='Annual_Price_Data', index=False)
        
        # Model-period matrix
        model_period_matrix.to_excel(writer, sheet_name='Model_Period_Matrix', index=True)
        
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

