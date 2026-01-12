import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from lasso_price_prediction import preprocess_features, get_feature_columns, get_trained_models

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
            entry_idx = quarters.index(entry_quarter)
            start_idx = max(0, entry_idx - 1)
            start_quarter_for_product = quarters[start_idx]
            
            exit_idx = quarters.index(exit_quarter)
            # End at exit quarter + 1, or until 2025 Q2 (whichever comes first)
            end_idx = min(len(quarters) - 1, exit_idx + 1)
            end_quarter_for_product = quarters[end_idx]
            
            lifecycle[idx] = {
                'entry_quarter': entry_quarter,
                'exit_quarter': exit_quarter,
                'start_quarter': start_quarter_for_product,  # One quarter before entry
                'end_quarter': end_quarter_for_product,  # One quarter after exit (or until 2025 Q2)
                'quarters_to_predict': quarters[start_idx:end_idx+1]
            }
    
    return lifecycle

def predict_with_error_feature(df, lifecycle, start_quarter='2020 Q1'):
    """
    Predict prices using Lasso models with previous quarter's prediction error as additional feature
    """
    feature_cols = get_feature_columns()
    quarters = get_sorted_quarter_columns(df)
    if start_quarter in quarters:
        quarters = quarters[quarters.index(start_quarter):]
    
    # Map column names to match what preprocess_features expects
    df_for_preprocess = df.copy()
    if 'RAM' in df_for_preprocess.columns and 'Ram Mem' not in df_for_preprocess.columns:
        df_for_preprocess['Ram Mem'] = df_for_preprocess['RAM']
    
    # Get base models (without error feature)
    print("Getting base Lasso models...")
    base_models, base_scalers, df_processed = get_trained_models(df_for_preprocess, start_quarter=start_quarter)
    
    # Store predictions and errors
    predictions = {}
    errors = {}  # {quarter: {product_idx: error}}
    
    # Extended feature columns (base features + previous error)
    extended_feature_cols = feature_cols + ['prev_quarter_error']
    
    # Train and predict sequentially: predict quarter i, then train quarter i+1 with errors from quarter i
    models_with_error = {}
    scalers_with_error = {}
    model_info = {}  # Store R² and model info for each quarter
    
    print("\nSequentially training models and predicting (with error feature)...")
    
    for i, quarter in enumerate(quarters):
        quarter_data = df_processed[df_processed[quarter].notna() & (df_processed[quarter] > 0)].copy()
        
        if len(quarter_data) < 10:
            continue
        
        if i == 0:
            # First quarter: use base model, no error feature
            models_with_error[quarter] = base_models[quarter]
            scalers_with_error[quarter] = base_scalers[quarter]
            
            # Calculate R² for first quarter (using base model)
            X = quarter_data[feature_cols].fillna(quarter_data[feature_cols].mean())
            X_scaled = base_scalers[quarter].transform(X)
            y = np.log(quarter_data[quarter])
            r2 = base_models[quarter].score(X_scaled, y)
            
            model_info[quarter] = {
                'n_samples': len(quarter_data),
                'r2_score': r2,
                'uses_error_feature': False
            }
            
            print(f"  {quarter}: Using base model (no error feature), R²={r2:.4f}")
            
            # Predict first quarter and calculate errors
            if quarter not in predictions:
                predictions[quarter] = {}
            if quarter not in errors:
                errors[quarter] = {}
                
            for idx in quarter_data.index:
                product_features = df_processed.loc[idx, feature_cols].fillna(df_processed[feature_cols].mean())
                product_features_scaled = base_scalers[quarter].transform([product_features])
                log_pred = base_models[quarter].predict(product_features_scaled)[0]
                pred_price = np.exp(log_pred)
                predictions[quarter][idx] = pred_price
                
                actual_price = df_processed.loc[idx, quarter]
                error = np.log(actual_price) - log_pred
                errors[quarter][idx] = error
        else:
            # Subsequent quarters: add previous quarter's error as feature
            prev_quarter = quarters[i - 1]
            
            if prev_quarter not in errors or len(errors[prev_quarter]) == 0:
                # If previous quarter has no errors, use base model
                models_with_error[quarter] = base_models[quarter]
                scalers_with_error[quarter] = base_scalers[quarter]
                
                # Calculate R² for this quarter (using base model)
                X = quarter_data[feature_cols].fillna(quarter_data[feature_cols].mean())
                X_scaled = base_scalers[quarter].transform(X)
                y = np.log(quarter_data[quarter])
                r2 = base_models[quarter].score(X_scaled, y)
                
                model_info[quarter] = {
                    'n_samples': len(quarter_data),
                    'r2_score': r2,
                    'uses_error_feature': False
                }
                
                print(f"  {quarter}: Using base model (no previous errors available), R²={r2:.4f}")
            else:
                # Prepare extended features with previous quarter's error
                X_base = quarter_data[feature_cols].copy()
                
                # Add previous quarter's error for products that have it
                prev_errors = []
                for idx in quarter_data.index:
                    if idx in errors[prev_quarter]:
                        prev_errors.append(errors[prev_quarter][idx])
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
                
                y = np.log(quarter_data[quarter])
                if y.isnull().any():
                    y = y.fillna(y.mean())
                
                # Standardize extended features
                scaler = StandardScaler()
                X_extended_scaled = scaler.fit_transform(X_extended)
                
                # Train Lasso with extended features
                lasso = LassoCV(cv=min(5, len(quarter_data)//2), random_state=42, max_iter=2000)
                lasso.fit(X_extended_scaled, y)
                
                models_with_error[quarter] = lasso
                scalers_with_error[quarter] = scaler
                
                r2 = lasso.score(X_extended_scaled, y)
                print(f"  {quarter}: Trained with error feature, {len(quarter_data)} samples, R²={r2:.4f}")
                
                # Record model info
                model_info[quarter] = {
                    'n_samples': len(quarter_data),
                    'r2_score': r2,
                    'uses_error_feature': True
                }
            
            # After training, predict this quarter and calculate errors for next quarter
            if quarter not in predictions:
                predictions[quarter] = {}
            if quarter not in errors:
                errors[quarter] = {}
                
            for idx in quarter_data.index:
                product_features = df_processed.loc[idx, feature_cols].fillna(df_processed[feature_cols].mean())
                
                # Check if model uses error feature
                expected_features = scalers_with_error[quarter].n_features_in_
                is_base_model = (expected_features == len(feature_cols))
                
                if is_base_model:
                    product_features_scaled = scalers_with_error[quarter].transform([product_features])
                    log_pred = models_with_error[quarter].predict(product_features_scaled)[0]
                else:
                    # Get previous error
                    prev_error = errors[prev_quarter].get(idx, 0.0)
                    product_features_extended = np.append(product_features.values, prev_error).reshape(1, -1)
                    product_features_scaled = scalers_with_error[quarter].transform(product_features_extended)
                    log_pred = models_with_error[quarter].predict(product_features_scaled)[0]
                
                pred_price = np.exp(log_pred)
                predictions[quarter][idx] = pred_price
                
                actual_price = df_processed.loc[idx, quarter]
                error = np.log(actual_price) - log_pred
                errors[quarter][idx] = error
    
    # Now predict for each product during its lifecycle
    # Process quarters sequentially to accumulate errors
    print("\nPredicting prices with error feature...")
    
    # Process products quarter by quarter to ensure errors are calculated in order
    for quarter_idx, quarter in enumerate(quarters):
        if quarter not in models_with_error:
            continue
        
        quarter_global_idx = quarters.index(quarter)
        
        # Find all products that should be predicted in this quarter
        products_for_quarter = []
        for product_idx, life_info in lifecycle.items():
            if quarter in life_info['quarters_to_predict']:
                products_for_quarter.append(product_idx)
        
        for product_idx in products_for_quarter:
            # Skip if already predicted (during training phase for products with actual prices)
            if quarter in predictions and product_idx in predictions[quarter]:
                continue
            
            # Get base features
            product_features = df_processed.loc[product_idx, feature_cols].fillna(df_processed[feature_cols].mean())
            
            # Check if this quarter's model uses error feature
            expected_features = scalers_with_error[quarter].n_features_in_
            is_base_model = (expected_features == len(feature_cols))
            
            if is_base_model:
                # Use base model (9 features)
                product_features_scaled = scalers_with_error[quarter].transform([product_features])
                log_pred = models_with_error[quarter].predict(product_features_scaled)[0]
            else:
                # Use model with error feature (10 features)
                prev_quarter = quarters[quarter_idx - 1]
                
                # Get previous quarter's error
                prev_error = 0.0
                if prev_quarter in errors and product_idx in errors[prev_quarter]:
                    prev_error = errors[prev_quarter][product_idx]
                elif prev_quarter in predictions and product_idx in predictions[prev_quarter]:
                    # Calculate error from prediction (even if no actual price)
                    if pd.notna(df_processed.loc[product_idx, prev_quarter]) and df_processed.loc[product_idx, prev_quarter] > 0:
                        # Has actual price: use actual error
                        actual_prev = df_processed.loc[product_idx, prev_quarter]
                        pred_prev = predictions[prev_quarter][product_idx]
                        prev_error = np.log(actual_prev) - np.log(pred_prev)
                    else:
                        # No actual price: use 0 (no correction from previous quarter)
                        prev_error = 0.0
                
                # Extended features - ensure correct column order
                product_features_array = product_features.values
                product_features_extended = np.append(product_features_array, prev_error)
                product_features_extended = product_features_extended.reshape(1, -1)
                
                product_features_scaled = scalers_with_error[quarter].transform(product_features_extended)
                log_pred = models_with_error[quarter].predict(product_features_scaled)[0]
            
            pred_price = np.exp(log_pred)
            
            if quarter not in predictions:
                predictions[quarter] = {}
            predictions[quarter][product_idx] = pred_price
            
            # Calculate error (if actual price exists) for next quarter to use
            if pd.notna(df_processed.loc[product_idx, quarter]) and df_processed.loc[product_idx, quarter] > 0:
                actual_price = df_processed.loc[product_idx, quarter]
                # Error in log space: ln(actual) - ln(predicted)
                error = np.log(actual_price) - log_pred
                
                if quarter not in errors:
                    errors[quarter] = {}
                errors[quarter][product_idx] = error
    
    # Convert to DataFrame
    id_cols = ['Company Name', 'Model Name']
    asin_cols = [col for col in df_processed.columns if 'ASIN' in col]
    id_cols = [col for col in id_cols if col in df_processed.columns] + asin_cols
    pred_df = df_processed[id_cols].copy()
    
    for quarter in quarters:
        col_name = f'{quarter}_predicted'
        pred_df[col_name] = np.nan
        
        if quarter in predictions:
            for product_idx, price in predictions[quarter].items():
                pred_df.loc[product_idx, col_name] = price
    
    return pred_df, model_info

def calculate_predicted_quarterly_jevons_index(df, quarter1_col, quarter2_col):
    """
    Calculate Jevons index between two predicted quarters
    """
    if quarter1_col not in df.columns or quarter2_col not in df.columns:
        return None, 0
    
    quarter1_prices = df[quarter1_col]
    quarter2_prices = df[quarter2_col]
    
    valid_mask = (quarter1_prices > 0) & (quarter2_prices > 0) & \
                 (~quarter1_prices.isna()) & (~quarter2_prices.isna())
    
    if valid_mask.sum() == 0:
        return None, 0
    
    q1_valid = quarter1_prices[valid_mask]
    q2_valid = quarter2_prices[valid_mask]
    
    log_price_ratios = np.log(q2_valid) - np.log(q1_valid)
    jevons_index = float(np.mean(log_price_ratios))
    
    return jevons_index, len(q1_valid)

def calculate_adjacent_predicted_quarterly_indices(df):
    """
    Calculate Jevons indices between adjacent predicted quarters
    """
    predicted_columns = [col for col in df.columns if '_predicted' in col]
    
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
    Main function: Predict prices with error feature and calculate Hedonic Jevons Index
    """
    print("Reading Dataset.xlsx...")
    df = pd.read_excel('../Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Find lifecycle for each product
    print("\nDetermining product lifecycles...")
    lifecycle = find_product_lifecycle(df, start_quarter='2020 Q1')
    print(f"Found lifecycle information for {len(lifecycle)} products")
    
    # Predict prices with error feature

    
    pred_df, model_info = predict_with_error_feature(df, lifecycle, start_quarter='2020 Q1')
    
    # Calculate Hedonic Jevons Indices

    
    print("\n=== Calculating Hedonic Jevons Indices (with error feature) ===")

    
    adjacent_results = calculate_adjacent_predicted_quarterly_indices(pred_df)

    
    

    
    # Create Model_Summary DataFrame

    
    model_summary_rows = []

    
    for quarter, info in model_info.items():

    
        model_summary_rows.append({

    
            'Quarter': quarter,

    
            'Samples': info['n_samples'],

    
            'R2_Score': info['r2_score'],

    
            'Uses_Error_Feature': info['uses_error_feature']

    
        })

    
    model_summary_df = pd.DataFrame(model_summary_rows)
 # Output to Excel
    output_file = 'Predicted_Jevons_Index_With_Error_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Predicted_Prices', index=False)

        adjacent_results.to_excel(writer, sheet_name='Adjacent Predicted Quarters', index=False)

        model_summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
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
    
    return pred_df, model_info, adjacent_results

if __name__ == "__main__":
    pred_df, model_info, adjacent_results = main()

