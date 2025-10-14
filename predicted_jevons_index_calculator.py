import pandas as pd
import numpy as np
import math
from itertools import combinations

def parse_predicted_quarter_column(col_name):
    """
    Parse predicted price column name, return (year, quarter)
    Example: "2020 Q1_predicted" -> (2020, 1)
    """
    try:
        if '_predicted' in col_name:
            base_name = col_name.replace('_predicted', '')
            parts = base_name.split()
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
    Example: 2020 Q1 -> 1, 2020 Q2 -> 2, ..., 2021 Q1 -> 5
    """
    base_year = 2020
    base_quarter = 1  # Starting from 2020 Q1
    
    # Calculate quarters from base point
    total_quarters = (year - base_year) * 4 + quarter - base_quarter + 1
    return total_quarters

def calculate_predicted_quarterly_jevons_index(df, quarter1_col, quarter2_col):
    """
    Calculate Jevons index between two predicted quarters
    
    Parameters:
    df: DataFrame containing predicted price data
    quarter1_col: Base period quarter column name (t-1)
    quarter2_col: Current period quarter column name (t)
    
    Returns:
    Jevons index value, number of valid products
    """
    
    if quarter1_col not in df.columns or quarter2_col not in df.columns:
        return None, 0
    
    # Get predicted price data for both quarters
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

def calculate_adjacent_predicted_quarterly_indices(df):
    """
    Calculate Jevons indices between adjacent predicted quarters
    """
    # Get all predicted price columns and sort them
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
    # Sort by chronological order
    quarter_data = []
    for col in predicted_columns:
        year, quarter = parse_predicted_quarter_column(col)
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
        
        jevons_index, n_products = calculate_predicted_quarterly_jevons_index(df, quarter1_col, quarter2_col)
        
        if jevons_index is not None:
            # Clean column names for display
            q1_display = quarter1_col.replace('_predicted', '')
            q2_display = quarter2_col.replace('_predicted', '')
            
            results.append({
                'Base Quarter': q1_display,
                'Current Quarter': q2_display,
                'Period': f"{q1_display} → {q2_display}",
                'Jevons Index': jevons_index,
                'Number of Products': n_products,
                'Price Change (%)': (jevons_index - 1) * 100
            })
            
            print(f"Predicted Jevons Index ({q1_display} → {q2_display}): {jevons_index:.4f} ({n_products} products)")
    
    return pd.DataFrame(results)

def calculate_all_predicted_quarterly_pairs(df):
    """
    Calculate Jevons indices for all predicted quarter pairs
    """
    # Get all predicted price columns and sort them
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
    # Sort by chronological order
    quarter_data = []
    for col in predicted_columns:
        year, quarter = parse_predicted_quarter_column(col)
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
            
            jevons_index, n_products = calculate_predicted_quarterly_jevons_index(df, quarter1_col, quarter2_col)
            
            if jevons_index is not None:
                # Clean column names for display
                q1_display = quarter1_col.replace('_predicted', '')
                q2_display = quarter2_col.replace('_predicted', '')
                
                # Calculate quarter interval
                year1, q1 = parse_predicted_quarter_column(quarter1_col)
                year2, q2 = parse_predicted_quarter_column(quarter2_col)
                quarters_apart = (year2 - year1) * 4 + (q2 - q1)
                
                results.append({
                    'Base Quarter': q1_display,
                    'Current Quarter': q2_display,
                    'Period': f"{q1_display} → {q2_display}",
                    'Jevons Index': jevons_index,
                    'Number of Products': n_products,
                    'Price Change (%)': (jevons_index - 1) * 100,
                    'Quarters Apart': quarters_apart
                })
    
    return pd.DataFrame(results)

def calculate_same_predicted_quarter_across_years(df):
    """
    Calculate Jevons indices for same predicted quarter across years (e.g., 2020 Q1 vs 2021 Q1)
    """
    # Group by quarter type
    quarters_by_type = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
    for col in predicted_columns:
        year, quarter = parse_predicted_quarter_column(col)
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
            
            jevons_index, n_products = calculate_predicted_quarterly_jevons_index(df, col1, col2)
            
            if jevons_index is not None:
                # Clean column names for display
                q1_display = col1.replace('_predicted', '')
                q2_display = col2.replace('_predicted', '')
                
                results.append({
                    'Quarter Type': quarter_type,
                    'Base Year': year1,
                    'Current Year': year2,
                    'Base Quarter': q1_display,
                    'Current Quarter': q2_display,
                    'Period': f"{q1_display} → {q2_display}",
                    'Jevons Index': jevons_index,
                    'Number of Products': n_products,
                    'Price Change (%)': (jevons_index - 1) * 100,
                    'Years Apart': year2 - year1
                })
    
    return pd.DataFrame(results)

def compare_actual_vs_predicted_jevons(df):
    """
    Compare Jevons indices between actual and predicted prices
    """
    # Get actual price columns and predicted price columns
    actual_columns = [col for col in df.columns if '_actual' in col and col != 'Company Name']
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
    comparison_results = []
    
    # Find matching actual and predicted price columns
    for pred_col in predicted_columns:
        actual_col = pred_col.replace('_predicted', '_actual')
        if actual_col in df.columns:
            quarter_name = pred_col.replace('_predicted', '')
            
            # Calculate quarterly Jevons index for actual prices (if previous quarter exists)
            prev_quarter_actual = None
            prev_quarter_predicted = None
            
            # Find previous quarter
            year, quarter = parse_predicted_quarter_column(pred_col)
            if quarter > 1:
                prev_year, prev_quarter_num = year, quarter - 1
            else:
                prev_year, prev_quarter_num = year - 1, 4
            
            prev_actual_col = f"{prev_year} Q{prev_quarter_num}_actual"
            prev_predicted_col = f"{prev_year} Q{prev_quarter_num}_predicted"
            
            if prev_actual_col in df.columns and prev_predicted_col in df.columns:
                # Calculate Jevons index for actual prices
                actual_jevons, actual_n = calculate_predicted_quarterly_jevons_index(
                    df, prev_actual_col, actual_col)
                
                # Calculate Jevons index for predicted prices
                predicted_jevons, predicted_n = calculate_predicted_quarterly_jevons_index(
                    df, prev_predicted_col, pred_col)
                
                if actual_jevons is not None and predicted_jevons is not None:
                    comparison_results.append({
                        'Quarter': quarter_name,
                        'Actual_Jevons': actual_jevons,
                        'Predicted_Jevons': predicted_jevons,
                        'Difference': predicted_jevons - actual_jevons,
                        'Actual_Products': actual_n,
                        'Predicted_Products': predicted_n,
                        'Actual_Change_Pct': (actual_jevons - 1) * 100,
                        'Predicted_Change_Pct': (predicted_jevons - 1) * 100
                    })
    
    return pd.DataFrame(comparison_results)

def main():
    """
    Main function: Read predicted data, calculate quarterly Jevons indices for predicted prices
    """
    print("Reading predicted price data...")
    df = pd.read_excel('Lasso_Price_Predictions.xlsx', sheet_name='Predictions')
    
    print(f"Dataset contains {len(df)} products")
    
    # Get all predicted price columns
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    print(f"Predicted price columns: {len(predicted_columns)} quarters")
    
    # Calculate adjacent predicted quarter Jevons indices
    print("\n=== Calculating Adjacent Predicted Quarter Jevons Indices ===")
    adjacent_results = calculate_adjacent_predicted_quarterly_indices(df)
    
    # Calculate all predicted quarter pair Jevons indices
    print(f"\n=== Calculating All Predicted Quarter Pair Jevons Indices (Total {len(predicted_columns)*(len(predicted_columns)-1)//2} pairs) ===")
    all_pairs_results = calculate_all_predicted_quarterly_pairs(df)
    print(f"Actually calculated {len(all_pairs_results)} valid quarter pairs")
    
    # Calculate same predicted quarter across years indices
    print("\n=== Calculating Same Predicted Quarter Across Years Jevons Indices ===")
    same_quarter_results = calculate_same_predicted_quarter_across_years(df)
    
    # Compare actual and predicted price Jevons indices
    print("\n=== Comparing Actual and Predicted Price Jevons Indices ===")
    comparison_results = compare_actual_vs_predicted_jevons(df)
    
    # Output to Excel file
    output_file = 'Predicted_Quarterly_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Adjacent predicted quarter results
        adjacent_results.to_excel(writer, sheet_name='Adjacent Predicted Quarters', index=False)
        
        # All predicted quarter pair results
        all_pairs_results.to_excel(writer, sheet_name='All Predicted Quarter Pairs', index=False)
        
        # Same predicted quarter across years results
        same_quarter_results.to_excel(writer, sheet_name='Same Predicted Quarter Across Years', index=False)
        
        # Actual vs predicted comparison
        if not comparison_results.empty:
            comparison_results.to_excel(writer, sheet_name='Actual vs Predicted Comparison', index=False)
        
        # Data summary
        summary_data = {
            'Metric': [
                'Total Products', 
                'Total Predicted Quarters', 
                'Adjacent Predicted Quarter Comparisons', 
                'Total Predicted Quarter Pair Comparisons',
                'Same Predicted Quarter Across Years Comparisons',
                'Actual vs Predicted Comparisons'
            ],
            'Value': [
                len(df), 
                len(predicted_columns),
                len(adjacent_results),
                len(all_pairs_results),
                len(same_quarter_results),
                len(comparison_results)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nPredicted price Jevons index results saved to {output_file}")
    print(f"Adjacent predicted quarter comparisons: {len(adjacent_results)}")
    print(f"All predicted quarter pair comparisons: {len(all_pairs_results)}")
    print(f"Same predicted quarter across years comparisons: {len(same_quarter_results)}")
    print(f"Actual vs predicted comparisons: {len(comparison_results)}")
    
    # Display some key results
    if not adjacent_results.empty:
        print("\n=== Adjacent Predicted Quarter Jevons Index Summary (First 10) ===")
        display_df = adjacent_results[['Period', 'Jevons Index', 'Price Change (%)', 'Number of Products']].head(10)
        print(display_df.to_string(index=False))
    
    if not comparison_results.empty:
        print("\n=== Actual vs Predicted Jevons Index Comparison (First 5) ===")
        display_df = comparison_results[['Quarter', 'Actual_Jevons', 'Predicted_Jevons', 'Difference']].head(5)
        print(display_df.to_string(index=False))
    
    return adjacent_results, all_pairs_results, same_quarter_results, comparison_results

if __name__ == "__main__":
    adjacent_results, all_pairs_results, same_quarter_results, comparison_results = main()
