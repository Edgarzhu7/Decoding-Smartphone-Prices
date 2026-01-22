import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import from Quarter folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Quarter'))
from lasso_price_prediction import preprocess_features, get_feature_columns

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
    
    # Map column names to match what preprocess_features expects
    if 'RAM' in annual_df.columns and 'Ram Mem' not in annual_df.columns:
        annual_df['Ram Mem'] = annual_df['RAM']
    
    return annual_df

def get_sorted_year_columns(df):
    """
    Get sorted year columns
    """
    year_columns = []
    for col in df.columns:
        try:
            year = int(col)
            year_columns.append(col)
        except:
            continue
    
    year_columns = sorted(year_columns, key=lambda x: int(x))
    return year_columns

def find_product_lifecycle(df, start_year='2020'):
    """
    Find entry and exit years for each product
    Entry: first year with actual price
    Exit: last year with actual price
    """
    years = get_sorted_year_columns(df)
    if start_year in years:
        years = years[years.index(start_year):]
    
    lifecycle = {}
    
    for idx, row in df.iterrows():
        # Find first year with price (entry)
        entry_year = None
        for year in years:
            if pd.notna(row[year]) and row[year] > 0:
                entry_year = year
                break
        
        # Find last year with price (exit)
        exit_year = None
        for year in reversed(years):
            if pd.notna(row[year]) and row[year] > 0:
                exit_year = year
                break
        
        if entry_year and exit_year:
            # Find entry year index
            entry_idx = years.index(entry_year)
            # Start from one year before entry
            start_idx = max(0, entry_idx - 1)
            start_year_for_product = years[start_idx]
            
            # End at exit year + 1, or until last available year (2025)
            exit_idx = years.index(exit_year)
            # Extend to one year after exit, but not beyond available years
            end_idx = min(len(years) - 1, exit_idx + 1)
            end_year_for_product = years[end_idx]
            
            lifecycle[idx] = {
                'entry_year': entry_year,
                'exit_year': exit_year,
                'start_year': start_year_for_product,  # One year before entry
                'end_year': end_year_for_product,  # One year after exit (or until 2025)
                'years_to_predict': years[start_idx:end_idx+1]
            }
    
    return lifecycle

def predict_prices_by_lifecycle(df, lifecycle, start_year='2020'):
    """
    Predict prices for each product only during its lifecycle
    Trains Lasso models independently but identically to lasso_price_prediction.py (annual version)
    
    Returns:
        pred_df: DataFrame with predicted prices
        model_info: Dictionary with model information (R², etc.) for each year
    """
    # Preprocess features (same as lasso_price_prediction.py)
    df_processed, processor_encoder = preprocess_features(df)
    feature_cols = get_feature_columns()
    
    years = get_sorted_year_columns(df)
    if start_year in years:
        years = years[years.index(start_year):]
    
    # Store predictions: {year: {product_idx: predicted_price}}
    predictions = {}
    
    # Train models for each year (identical logic to lasso_price_prediction.py)
    models = {}
    scalers = {}
    model_info = {}  # Store R² and model info for each year
    
    print("Training Lasso models for each year (identical to lasso_price_prediction.py)...")
    for year in years:
        # Get samples with price data for this year (same logic)
        year_data = df_processed[df_processed[year].notna() & (df_processed[year] > 0)].copy()
        
        if len(year_data) < 10:  # Need at least 10 samples
            continue
        
        # Prepare features and target variable (same as lasso_price_prediction.py)
        X = year_data[feature_cols]
        y = np.log(year_data[year])  # log price
        
        # Check data quality (same as lasso_price_prediction.py)
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
        
        # Standardize features (same as lasso_price_prediction.py)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso regression (same parameters as lasso_price_prediction.py)
        lasso = LassoCV(cv=min(5, len(year_data)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        models[year] = lasso
        scalers[year] = scaler
        
        r2_score = lasso.score(X_scaled, y)
        print(f"  Trained model for {year}: {len(year_data)} samples, R²={r2_score:.4f}")
        
        # Store model info
        model_info[year] = {
            'n_samples': len(year_data),
            'r2_score': r2_score,
            'alpha': lasso.alpha_,
            'n_features_selected': np.sum(lasso.coef_ != 0)
        }
    
    # Now predict for each product during its lifecycle
    print("\nPredicting prices for each product during its lifecycle...")
    for product_idx, life_info in lifecycle.items():
        years_to_predict = life_info['years_to_predict']
        
        for year in years_to_predict:
            if year not in models:
                continue
            
            # Get product features (same as lasso_price_prediction.py)
            product_features = df_processed.loc[product_idx, feature_cols].fillna(df_processed[feature_cols].mean())
            product_features_scaled = scalers[year].transform([product_features])
            
            # Predict (same as lasso_price_prediction.py)
            log_pred = models[year].predict(product_features_scaled)[0]
            pred_price = np.exp(log_pred)  # Convert back to price
            
            if year not in predictions:
                predictions[year] = {}
            predictions[year][product_idx] = pred_price
    
    # Convert to DataFrame
    id_cols = ['Company Name', 'Model Name']
    asin_cols = [col for col in df_processed.columns if 'ASIN' in col]
    id_cols = [col for col in id_cols if col in df_processed.columns] + asin_cols
    pred_df = df_processed[id_cols].copy()
    
    for year in years:
        col_name = f'{year}_predicted'
        pred_df[col_name] = np.nan
        
        if year in predictions:
            for product_idx, price in predictions[year].items():
                pred_df.loc[product_idx, col_name] = price
    
    return pred_df, model_info

def calculate_predicted_annual_jevons_index(df, year1_col, year2_col):
    """
    Calculate Jevons index between two predicted years
    Only uses products that have predictions in both years
    """
    if year1_col not in df.columns or year2_col not in df.columns:
        return None, 0
    
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

def calculate_adjacent_predicted_annual_indices(df):
    """
    Calculate Jevons indices between adjacent predicted years
    """
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
    # Sort by chronological order
    year_data = []
    for col in predicted_columns:
        year_str = col.replace('_predicted', '')
        try:
            year = int(year_str)
            year_data.append((year, col))
        except:
            continue
    
    year_data.sort(key=lambda x: x[0])
    sorted_columns = [item[1] for item in year_data]
    
    results = []
    
    for i in range(len(sorted_columns) - 1):
        year1_col = sorted_columns[i]
        year2_col = sorted_columns[i + 1]
        
        jevons_index, n_products = calculate_predicted_annual_jevons_index(df, year1_col, year2_col)
        
        if jevons_index is not None:
            y1_display = year1_col.replace('_predicted', '')
            y2_display = year2_col.replace('_predicted', '')
            
            results.append({
                'Base Year': y1_display,
                'Current Year': y2_display,
                'Period': f"{y1_display} → {y2_display}",
                'Jevons Index': jevons_index,
                'Number of Products': n_products,
                'Price Change (%)': jevons_index * 100
            })
    
    return pd.DataFrame(results)

def main():
    """
    Main function: Aggregate to annual, predict prices by product lifecycle and calculate Hedonic Jevons Index
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
    
    # Find lifecycle for each product
    print("\nDetermining product lifecycles...")
    lifecycle = find_product_lifecycle(annual_df, start_year='2020')
    print(f"Found lifecycle information for {len(lifecycle)} products")
    
    # Show some examples
    print("\nExample lifecycles:")
    for i, (idx, life) in enumerate(list(lifecycle.items())[:5]):
        print(f"  Product {idx}: Entry={life['entry_year']}, Exit={life['exit_year']}, "
              f"Predict from {life['start_year']} to {life['end_year']} ({len(life['years_to_predict'])} years)")
    
    # Predict prices by lifecycle
    pred_df, model_info = predict_prices_by_lifecycle(annual_df, lifecycle, start_year='2020')
    
    # Calculate Hedonic Jevons Indices
    print("\n=== Calculating Hedonic Jevons Indices ===")
    adjacent_results = calculate_adjacent_predicted_annual_indices(pred_df)
    
    # Create Model_Summary DataFrame
    model_summary_rows = []
    for year, info in model_info.items():
        model_summary_rows.append({
            'Year': year,
            'Samples': info['n_samples'],
            'R2_Score': info['r2_score'],
            'Alpha': info['alpha'],
            'Features_Selected': info['n_features_selected']
        })
    model_summary_df = pd.DataFrame(model_summary_rows)
    
    # Output to Excel
    output_file = 'Predicted_Annual_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Predicted_Prices', index=False)
        adjacent_results.to_excel(writer, sheet_name='Adjacent Predicted Years', index=False)
        model_summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
        
        # Summary
        summary_data = {
            'Metric': [
                'Total Products',
                'Products with Lifecycle Info',
                'Adjacent Year Comparisons'
            ],
            'Value': [
                len(annual_df),
                len(lifecycle),
                len(adjacent_results)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Adjacent year comparisons: {len(adjacent_results)}")
    
    if not adjacent_results.empty:
        print("\n=== Adjacent Year Hedonic Jevons Index Summary ===")
        print(adjacent_results[['Period', 'Jevons Index', 'Price Change (%)', 'Number of Products']].to_string(index=False))
        
        cumulative = adjacent_results['Jevons Index'].sum()
        print(f"\nCumulative Hedonic Jevons Index: {cumulative:.6f}")
        print(f"Cumulative price change: {cumulative * 100:.2f}%")
    
    return pred_df, adjacent_results

if __name__ == "__main__":
    pred_df, adjacent_results = main()

