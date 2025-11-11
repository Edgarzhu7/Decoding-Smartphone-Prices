import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from lasso_price_prediction import preprocess_features, get_feature_columns

def get_sorted_quarter_columns(df):
    """Get sorted quarter columns"""
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    quarter_columns = sorted(quarter_columns, key=lambda x: (int(x.split()[0]), int(x.split()[1][1:])))
    return quarter_columns

def find_product_lifecycle(df, start_quarter='2020 Q1'):
    """
    Find entry and exit quarters for each product
    Entry: first quarter with actual price
    Exit: last quarter with actual price
    """
    quarters = get_sorted_quarter_columns(df)
    if start_quarter in quarters:
        quarters = quarters[quarters.index(start_quarter):]
    
    lifecycle = {}
    
    for idx, row in df.iterrows():
        # Find first quarter with price (entry)
        entry_quarter = None
        for q in quarters:
            if pd.notna(row[q]) and row[q] > 0:
                entry_quarter = q
                break
        
        # Find last quarter with price (exit)
        exit_quarter = None
        for q in reversed(quarters):
            if pd.notna(row[q]) and row[q] > 0:
                exit_quarter = q
                break
        
        if entry_quarter and exit_quarter:
            # Find entry quarter index
            entry_idx = quarters.index(entry_quarter)
            # Start from one quarter before entry
            start_idx = max(0, entry_idx - 1)
            start_quarter_for_product = quarters[start_idx]
            
            # End at exit quarter
            exit_idx = quarters.index(exit_quarter)
            end_quarter_for_product = quarters[exit_idx]
            
            lifecycle[idx] = {
                'entry_quarter': entry_quarter,
                'exit_quarter': exit_quarter,
                'start_quarter': start_quarter_for_product,  # One quarter before entry
                'end_quarter': end_quarter_for_product,
                'quarters_to_predict': quarters[start_idx:exit_idx+1]
            }
    
    return lifecycle

def predict_prices_by_lifecycle(df, lifecycle, start_quarter='2020 Q1'):
    """
    Predict prices for each product only during its lifecycle
    Trains Lasso models independently but identically to lasso_price_prediction.py
    """
    # Preprocess features (same as lasso_price_prediction.py)
    df_processed, processor_encoder = preprocess_features(df)
    feature_cols = get_feature_columns()
    
    quarters = get_sorted_quarter_columns(df)
    if start_quarter in quarters:
        quarters = quarters[quarters.index(start_quarter):]
    
    # Store predictions: {quarter: {product_idx: predicted_price}}
    predictions = {}
    
    # Train models for each quarter (identical to lasso_price_prediction.py)
    models = {}
    scalers = {}
    
    print("Training Lasso models for each quarter (identical to lasso_price_prediction.py)...")
    for quarter in quarters:
        # Get samples with price data for this quarter (same logic)
        quarter_data = df_processed[df_processed[quarter].notna() & (df_processed[quarter] > 0)].copy()
        
        if len(quarter_data) < 10:  # Need at least 10 samples
            continue
        
        # Prepare features and target variable (same as lasso_price_prediction.py)
        X = quarter_data[feature_cols]
        y = np.log(quarter_data[quarter])  # log price
        
        # Check data quality (same as lasso_price_prediction.py)
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
        
        # Standardize features (same as lasso_price_prediction.py)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso regression (same parameters as lasso_price_prediction.py)
        lasso = LassoCV(cv=min(5, len(quarter_data)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        models[quarter] = lasso
        scalers[quarter] = scaler
        
        r2_score = lasso.score(X_scaled, y)
        print(f"  Trained model for {quarter}: {len(quarter_data)} samples, R²={r2_score:.4f}")
    
    # Now predict for each product during its lifecycle
    print("\nPredicting prices for each product during its lifecycle...")
    for product_idx, life_info in lifecycle.items():
        quarters_to_predict = life_info['quarters_to_predict']
        
        for quarter in quarters_to_predict:
            if quarter not in models:
                continue
            
            # Get product features (same as lasso_price_prediction.py)
            product_features = df_processed.loc[product_idx, feature_cols].fillna(df_processed[feature_cols].mean())
            product_features_scaled = scalers[quarter].transform([product_features])
            
            # Predict (same as lasso_price_prediction.py)
            log_pred = models[quarter].predict(product_features_scaled)[0]
            pred_price = np.exp(log_pred)  # Convert back to price
            
            if quarter not in predictions:
                predictions[quarter] = {}
            predictions[quarter][product_idx] = pred_price
    
    # Convert to DataFrame
    pred_df = df_processed[['Company Name', 'Model Name', 'ASIN']].copy()
    
    for quarter in quarters:
        col_name = f'{quarter}_predicted'
        pred_df[col_name] = np.nan
        
        if quarter in predictions:
            for product_idx, price in predictions[quarter].items():
                pred_df.loc[product_idx, col_name] = price
    
    return pred_df

def calculate_predicted_quarterly_jevons_index(df, quarter1_col, quarter2_col):
    """
    Calculate Jevons index between two predicted quarters
    Only uses products that have predictions in both quarters
    """
    if quarter1_col not in df.columns or quarter2_col not in df.columns:
        return None, 0
    
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
    
    # Calculate Jevons index (mean log difference, without exp)
    jevons_index = float(np.mean(log_price_ratios))
    
    return jevons_index, len(q1_valid)

def calculate_adjacent_predicted_quarterly_indices(df):
    """
    Calculate Jevons indices between adjacent predicted quarters
    """
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
    # Sort by chronological order
    quarter_data = []
    for col in predicted_columns:
        parts = col.replace('_predicted', '').split()
        if len(parts) >= 2 and 'Q' in parts[1]:
            year = int(parts[0])
            quarter = int(parts[1].replace('Q', ''))
            order = (year - 2020) * 4 + quarter
            quarter_data.append((order, col, year, quarter))
    
    quarter_data.sort(key=lambda x: x[0])
    sorted_columns = [item[1] for item in quarter_data]
    
    results = []
    
    for i in range(len(sorted_columns) - 1):
        quarter1_col = sorted_columns[i]
        quarter2_col = sorted_columns[i + 1]
        
        jevons_index, n_products = calculate_predicted_quarterly_jevons_index(df, quarter1_col, quarter2_col)
        
        if jevons_index is not None:
            q1_display = quarter1_col.replace('_predicted', '')
            q2_display = quarter2_col.replace('_predicted', '')
            
            results.append({
                'Base Quarter': q1_display,
                'Current Quarter': q2_display,
                'Period': f"{q1_display} → {q2_display}",
                'Jevons Index': jevons_index,
                'Number of Products': n_products,
                'Price Change (%)': jevons_index * 100
            })
    
    return pd.DataFrame(results)

def main():
    """
    Main function: Predict prices by product lifecycle and calculate Hedonic Jevons Index
    """
    print("Reading Dataset.xlsx...")
    df = pd.read_excel('Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Find lifecycle for each product
    print("\nDetermining product lifecycles...")
    lifecycle = find_product_lifecycle(df, start_quarter='2020 Q1')
    print(f"Found lifecycle information for {len(lifecycle)} products")
    
    # Show some examples
    print("\nExample lifecycles:")
    for i, (idx, life) in enumerate(list(lifecycle.items())[:5]):
        print(f"  Product {idx}: Entry={life['entry_quarter']}, Exit={life['exit_quarter']}, "
              f"Predict from {life['start_quarter']} to {life['end_quarter']} ({len(life['quarters_to_predict'])} quarters)")
    
    # Predict prices by lifecycle
    pred_df = predict_prices_by_lifecycle(df, lifecycle, start_quarter='2020 Q1')
    
    # Calculate Hedonic Jevons Indices
    print("\n=== Calculating Hedonic Jevons Indices ===")
    adjacent_results = calculate_adjacent_predicted_quarterly_indices(pred_df)
    
    # Output to Excel
    output_file = 'Predicted_Quarterly_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Predicted_Prices', index=False)
        adjacent_results.to_excel(writer, sheet_name='Adjacent Predicted Quarters', index=False)
        
        # Summary
        summary_data = {
            'Metric': [
                'Total Products',
                'Products with Lifecycle Info',
                'Adjacent Quarter Comparisons'
            ],
            'Value': [
                len(df),
                len(lifecycle),
                len(adjacent_results)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Adjacent quarter comparisons: {len(adjacent_results)}")
    
    if not adjacent_results.empty:
        print("\n=== Adjacent Quarter Hedonic Jevons Index Summary ===")
        print(adjacent_results[['Period', 'Jevons Index', 'Price Change (%)', 'Number of Products']].to_string(index=False))
        
        cumulative = adjacent_results['Jevons Index'].sum()
        print(f"\nCumulative Hedonic Jevons Index: {cumulative:.6f}")
        print(f"Cumulative price change: {cumulative * 100:.2f}%")
    
    return pred_df, adjacent_results

if __name__ == "__main__":
    pred_df, adjacent_results = main()
