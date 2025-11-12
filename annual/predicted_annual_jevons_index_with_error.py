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
            entry_idx = years.index(entry_year)
            start_idx = max(0, entry_idx - 1)
            start_year_for_product = years[start_idx]
            
            exit_idx = years.index(exit_year)
            end_year_for_product = years[exit_idx]
            
            lifecycle[idx] = {
                'entry_year': entry_year,
                'exit_year': exit_year,
                'start_year': start_year_for_product,
                'end_year': end_year_for_product,
                'years_to_predict': years[start_idx:exit_idx+1]
            }
    
    return lifecycle

def get_base_annual_models(df, start_year='2020'):
    """
    Get base Lasso models for each year (without error feature)
    This function trains models for annual data
    """
    df_processed, _ = preprocess_features(df)
    years = get_sorted_year_columns(df)
    if start_year in years:
        years = years[years.index(start_year):]
    
    feature_cols = get_feature_columns()
    models = {}
    scalers = {}
    
    for year in years:
        year_data = df_processed[df_processed[year].notna() & (df_processed[year] > 0)].copy()
        
        if len(year_data) < 10:
            continue
        
        X = year_data[feature_cols]
        y = np.log(year_data[year])
        
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        if y.isnull().any():
            y = y.fillna(y.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LassoCV(cv=min(5, len(year_data)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        models[year] = lasso
        scalers[year] = scaler
    
    return models, scalers, df_processed

def predict_with_error_feature(df, lifecycle, start_year='2020'):
    """
    Predict prices using Lasso models with previous year's prediction error as additional feature
    """
    feature_cols = get_feature_columns()
    years = get_sorted_year_columns(df)
    if start_year in years:
        years = years[years.index(start_year):]
    
    # Get base models (without error feature)
    print("Getting base Lasso models for annual data...")
    base_models, base_scalers, df_processed = get_base_annual_models(df, start_year=start_year)
    
    # Store predictions and errors
    predictions = {}
    errors = {}  # {year: {product_idx: error}}
    
    # Train and predict sequentially: predict year i, then train year i+1 with errors from year i
    models_with_error = {}
    scalers_with_error = {}
    
    print("\nSequentially training models and predicting (with error feature)...")
    
    for i, year in enumerate(years):
        year_data = df_processed[df_processed[year].notna() & (df_processed[year] > 0)].copy()
        
        if len(year_data) < 10:
            continue
        
        if i == 0:
            # First year: use base model, no error feature
            models_with_error[year] = base_models[year]
            scalers_with_error[year] = base_scalers[year]
            print(f"  {year}: Using base model (no error feature)")
            
            # Predict first year and calculate errors
            if year not in predictions:
                predictions[year] = {}
            if year not in errors:
                errors[year] = {}
                
            for idx in year_data.index:
                product_features = df_processed.loc[idx, feature_cols].fillna(df_processed[feature_cols].mean())
                product_features_scaled = base_scalers[year].transform([product_features])
                log_pred = base_models[year].predict(product_features_scaled)[0]
                pred_price = np.exp(log_pred)
                predictions[year][idx] = pred_price
                
                actual_price = df_processed.loc[idx, year]
                error = np.log(actual_price) - log_pred
                errors[year][idx] = error
        else:
            # Subsequent years: add previous year's error as feature
            prev_year = years[i - 1]
            
            if prev_year not in errors or len(errors[prev_year]) == 0:
                # If previous year has no errors, use base model
                models_with_error[year] = base_models[year]
                scalers_with_error[year] = base_scalers[year]
                print(f"  {year}: Using base model (no previous errors available)")
            else:
                # Prepare extended features with previous year's error
                X_base = year_data[feature_cols].copy()
                
                # Add previous year's error for products that have it
                prev_errors = []
                for idx in year_data.index:
                    if idx in errors[prev_year]:
                        prev_errors.append(errors[prev_year][idx])
                    else:
                        # If no previous error, use 0 (no correction)
                        prev_errors.append(0.0)
                
                # Fill missing values in base features first
                if X_base.isnull().any().any():
                    X_base = X_base.fillna(X_base.mean())
                
                # Create extended features array: [base_features, prev_error]
                X_extended = np.column_stack([X_base.values, prev_errors])
                
                # Fill any remaining NaN values
                if np.isnan(X_extended).any():
                    col_means = np.nanmean(X_extended, axis=0)
                    nan_mask = np.isnan(X_extended)
                    X_extended[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                
                y = np.log(year_data[year])
                if y.isnull().any():
                    y = y.fillna(y.mean())
                
                # Standardize extended features
                scaler = StandardScaler()
                X_extended_scaled = scaler.fit_transform(X_extended)
                
                # Train Lasso with extended features
                lasso = LassoCV(cv=min(5, len(year_data)//2), random_state=42, max_iter=2000)
                lasso.fit(X_extended_scaled, y)
                
                models_with_error[year] = lasso
                scalers_with_error[year] = scaler
                
                r2 = lasso.score(X_extended_scaled, y)
                print(f"  {year}: Trained with error feature, {len(year_data)} samples, R²={r2:.4f}")
            
            # After training, predict this year and calculate errors for next year
            if year not in predictions:
                predictions[year] = {}
            if year not in errors:
                errors[year] = {}
                
            for idx in year_data.index:
                product_features = df_processed.loc[idx, feature_cols].fillna(df_processed[feature_cols].mean())
                
                # Check if model uses error feature
                expected_features = scalers_with_error[year].n_features_in_
                is_base_model = (expected_features == len(feature_cols))
                
                if is_base_model:
                    product_features_scaled = scalers_with_error[year].transform([product_features])
                    log_pred = models_with_error[year].predict(product_features_scaled)[0]
                else:
                    # Get previous error
                    prev_error = errors[prev_year].get(idx, 0.0)
                    product_features_extended = np.append(product_features.values, prev_error).reshape(1, -1)
                    product_features_scaled = scalers_with_error[year].transform(product_features_extended)
                    log_pred = models_with_error[year].predict(product_features_scaled)[0]
                
                pred_price = np.exp(log_pred)
                predictions[year][idx] = pred_price
                
                actual_price = df_processed.loc[idx, year]
                error = np.log(actual_price) - log_pred
                errors[year][idx] = error
    
    # Now predict for each product during its lifecycle
    # Process years sequentially to accumulate errors
    print("\nPredicting prices with error feature...")
    
    # Process products year by year to ensure errors are calculated in order
    for year_idx, year in enumerate(years):
        if year not in models_with_error:
            continue
        
        # Find all products that should be predicted in this year
        products_for_year = []
        for product_idx, life_info in lifecycle.items():
            if year in life_info['years_to_predict']:
                products_for_year.append(product_idx)
        
        for product_idx in products_for_year:
            # Skip if already predicted (during training phase for products with actual prices)
            if year in predictions and product_idx in predictions[year]:
                continue
            
            # Get base features
            product_features = df_processed.loc[product_idx, feature_cols].fillna(df_processed[feature_cols].mean())
            
            # Check if this year's model uses error feature
            expected_features = scalers_with_error[year].n_features_in_
            is_base_model = (expected_features == len(feature_cols))
            
            if is_base_model:
                # Use base model (9 features)
                product_features_scaled = scalers_with_error[year].transform([product_features])
                log_pred = models_with_error[year].predict(product_features_scaled)[0]
            else:
                # Use model with error feature (10 features)
                prev_year = years[year_idx - 1]
                
                # Get previous year's error
                prev_error = 0.0
                if prev_year in errors and product_idx in errors[prev_year]:
                    prev_error = errors[prev_year][product_idx]
                elif prev_year in predictions and product_idx in predictions[prev_year]:
                    # Calculate error from prediction (even if no actual price)
                    if pd.notna(df_processed.loc[product_idx, prev_year]) and df_processed.loc[product_idx, prev_year] > 0:
                        # Has actual price: use actual error
                        actual_prev = df_processed.loc[product_idx, prev_year]
                        pred_prev = predictions[prev_year][product_idx]
                        prev_error = np.log(actual_prev) - np.log(pred_prev)
                    else:
                        # No actual price: use 0 (no correction from previous year)
                        prev_error = 0.0
                
                # Extended features - ensure correct column order
                product_features_array = product_features.values
                product_features_extended = np.append(product_features_array, prev_error)
                product_features_extended = product_features_extended.reshape(1, -1)
                
                product_features_scaled = scalers_with_error[year].transform(product_features_extended)
                log_pred = models_with_error[year].predict(product_features_scaled)[0]
            
            pred_price = np.exp(log_pred)
            
            if year not in predictions:
                predictions[year] = {}
            predictions[year][product_idx] = pred_price
            
            # Calculate error (if actual price exists) for next year to use
            if pd.notna(df_processed.loc[product_idx, year]) and df_processed.loc[product_idx, year] > 0:
                actual_price = df_processed.loc[product_idx, year]
                # Error in log space: ln(actual) - ln(predicted)
                error = np.log(actual_price) - log_pred
                
                if year not in errors:
                    errors[year] = {}
                errors[year][product_idx] = error
    
    # Convert to DataFrame
    pred_df = df_processed[['Company Name', 'Model Name', 'ASIN']].copy()
    
    for year in years:
        col_name = f'{year}_predicted'
        pred_df[col_name] = np.nan
        
        if year in predictions:
            for product_idx, price in predictions[year].items():
                pred_df.loc[product_idx, col_name] = price
    
    return pred_df

def calculate_predicted_annual_jevons_index(df, year1_col, year2_col):
    """
    Calculate Jevons index between two predicted years
    """
    if year1_col not in df.columns or year2_col not in df.columns:
        return None, 0
    
    year1_prices = df[year1_col]
    year2_prices = df[year2_col]
    
    valid_mask = (year1_prices > 0) & (year2_prices > 0) & \
                 (~year1_prices.isna()) & (~year2_prices.isna())
    
    if valid_mask.sum() == 0:
        return None, 0
    
    y1_valid = year1_prices[valid_mask]
    y2_valid = year2_prices[valid_mask]
    
    log_price_ratios = np.log(y2_valid) - np.log(y1_valid)
    jevons_index = float(np.mean(log_price_ratios))
    
    return jevons_index, len(y1_valid)

def calculate_adjacent_predicted_annual_indices(df):
    """
    Calculate Jevons indices between adjacent predicted years
    """
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
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
    Main function: Aggregate to annual, predict prices with error feature and calculate Hedonic Jevons Index
    """
    print("Reading Dataset.xlsx...")
    df = pd.read_excel('../Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Aggregate quarterly data to annual data
    print("\nAggregating quarterly data to annual averages...")
    annual_df = aggregate_quarters_to_years(df)
    
    # Find lifecycle for each product
    print("\nDetermining product lifecycles...")
    lifecycle = find_product_lifecycle(annual_df, start_year='2020')
    print(f"Found lifecycle information for {len(lifecycle)} products")
    
    # Predict prices with error feature
    pred_df = predict_with_error_feature(annual_df, lifecycle, start_year='2020')
    
    # Calculate Hedonic Jevons Indices
    print("\n=== Calculating Hedonic Jevons Indices (with error feature) ===")
    adjacent_results = calculate_adjacent_predicted_annual_indices(pred_df)
    
    # Output to Excel
    output_file = 'Predicted_Annual_Jevons_Index_With_Error_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Predicted_Prices', index=False)
        adjacent_results.to_excel(writer, sheet_name='Adjacent Predicted Years', index=False)
        
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

