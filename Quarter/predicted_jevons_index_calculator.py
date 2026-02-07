import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import statsmodels.api as sm
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
            
            # End at exit quarter + 1, or until 2025 Q2 (whichever comes first)
            exit_idx = quarters.index(exit_quarter)
            # Extend to one quarter after exit, but not beyond available quarters
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

def predict_prices_by_lifecycle(df, lifecycle, start_quarter='2020 Q1'):
    """
    Predict prices for each product only during its lifecycle
    Trains Lasso models independently but identically to lasso_price_prediction.py
    
    Returns:
        pred_df: DataFrame with predicted prices
        model_info: Dictionary with model information (R², etc.) for each quarter
    """
    # Map column names to match what preprocess_features expects
    df_for_preprocess = df.copy()
    if 'RAM' in df_for_preprocess.columns and 'Ram Mem' not in df_for_preprocess.columns:
        df_for_preprocess['Ram Mem'] = df_for_preprocess['RAM']
    
    # Preprocess features (same as lasso_price_prediction.py)
    df_processed, processor_encoder = preprocess_features(df_for_preprocess)
    feature_cols = get_feature_columns()
    
    quarters = get_sorted_quarter_columns(df)
    if start_quarter in quarters:
        quarters = quarters[quarters.index(start_quarter):]
    
    # Store predictions: {quarter: {product_idx: predicted_price}}
    predictions = {}
    
    # Train models for each quarter (identical to lasso_price_prediction.py)
    models = {}
    scalers = {}
    model_info = {}  # Store R² and model info for each quarter
    coef_rows = []  # Store coefficients for each quarter
    
    def bootstrap_lasso_coefficients(X_scaled, y, alpha, n_bootstraps=500, random_state=42):
        """
        Calculate bootstrap confidence intervals for Lasso coefficients
        Uses post-selection inference: refit OLS on selected features
        """
        np.random.seed(random_state)
        n_features = X_scaled.shape[1]
        lasso = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
        boot_coefs = []
        
        for _ in range(n_bootstraps):
            # Generate bootstrap resample
            X_resampled, y_resampled = resample(X_scaled, y)
            
            # Fit LASSO on bootstrap sample
            lasso.fit(X_resampled, y_resampled)
            selected_features = np.where(lasso.coef_ != 0)[0]
            
            if len(selected_features) > 0:
                # Refit OLS on selected features
                X_selected = X_resampled[:, selected_features]
                X_selected = sm.add_constant(X_selected)  # Add intercept
                ols = sm.OLS(y_resampled, X_selected).fit()
                coef_full = np.zeros(n_features)  # Include zeros for non-selected
                coef_full[selected_features] = ols.params[1:]  # Skip intercept
                boot_coefs.append(coef_full)
            else:
                # If no features selected, use zero coefficients
                boot_coefs.append(np.zeros(n_features))
        
        if len(boot_coefs) == 0:
            return None, None
        
        boot_coefs = np.array(boot_coefs)
        # Calculate confidence intervals (2.5th and 97.5th percentiles for 95% CI)
        conf_intervals = np.percentile(boot_coefs, [2.5, 97.5], axis=0)
        return boot_coefs, conf_intervals
    
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
        
        # Calculate bootstrap confidence intervals
        print(f"    Calculating bootstrap confidence intervals (n_bootstrap=500)...")
        boot_coefs, conf_intervals = bootstrap_lasso_coefficients(
            X_scaled, y.values, lasso.alpha_, n_bootstraps=500, random_state=42
        )
        
        # Store model info
        model_info[quarter] = {
            'n_samples': len(quarter_data),
            'r2_score': r2_score,
            'alpha': lasso.alpha_,
            'n_features_selected': np.sum(lasso.coef_ != 0)
        }
        
        # Store coefficients with confidence intervals
        for j, f in enumerate(feature_cols):
            coef_value = lasso.coef_[j]
            if conf_intervals is not None:
                lower_bound = conf_intervals[0, j]
                upper_bound = conf_intervals[1, j]
            else:
                lower_bound = np.nan
                upper_bound = np.nan
            
            coef_rows.append({
                'Quarter': quarter,
                'Feature': f,
                'Coefficient': coef_value,
                'Abs_Coefficient': abs(coef_value),
                'CI_Lower': lower_bound,
                'CI_Upper': upper_bound
            })
    
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
    
    coef_df = pd.DataFrame(coef_rows) if coef_rows else pd.DataFrame()
    return pred_df, model_info, coef_df

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
    df = pd.read_excel('../Dataset.xlsx')
    
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
    pred_df, model_info, coef_df = predict_prices_by_lifecycle(df, lifecycle, start_quarter='2020 Q1')
    
    # Calculate Hedonic Jevons Indices
    print("\n=== Calculating Hedonic Jevons Indices ===")
    adjacent_results = calculate_adjacent_predicted_quarterly_indices(pred_df)
    
    # Create Model_Summary DataFrame
    model_summary_rows = []
    for quarter, info in model_info.items():
        model_summary_rows.append({
            'Quarter': quarter,
            'Samples': info['n_samples'],
            'R2_Score': info['r2_score'],
            'Alpha': info['alpha'],
            'Features_Selected': info['n_features_selected']
        })
    model_summary_df = pd.DataFrame(model_summary_rows)
    
    # Output to Excel
    output_file = 'Predicted_Quarterly_Jevons_Index_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Predicted_Prices', index=False)
        adjacent_results.to_excel(writer, sheet_name='Adjacent Predicted Quarters', index=False)
        model_summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
        if not coef_df.empty:
            coef_df.to_excel(writer, sheet_name='Coefficients', index=False)
        
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
