import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List
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
    Returns lifecycle dict with start and end quarters for prediction
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
            # Start from one quarter before entry
            start_idx = max(0, entry_idx - 1)
            start_quarter_for_product = quarters[start_idx]
            
            # End at exit quarter + 1, or until 2025 Q2 (whichever comes first)
            exit_idx = quarters.index(exit_quarter)
            end_idx = min(len(quarters) - 1, exit_idx + 1)
            end_quarter_for_product = quarters[end_idx]
            
            lifecycle[idx] = {
                'entry_quarter': entry_quarter,
                'exit_quarter': exit_quarter,
                'start_quarter': start_quarter_for_product,  # One quarter before entry
                'end_quarter': end_quarter_for_product,  # One quarter after exit (or until 2025 Q2)
                'start_idx': start_idx,
                'end_idx': end_idx,
                'quarters_to_predict': quarters[start_idx:end_idx+1]
            }
    
    return lifecycle


def run_time_dummy_models(df, start_quarter='2020 Q1'):
    """
    Run time dummy models: pool consecutive periods together, add time dummy variable,
    train a single OLS (unpenalized linear regression) model, and predict prices for both periods
    """
    # Map column names to match what preprocess_features expects
    df_for_preprocess = df.copy()
    if 'RAM' in df_for_preprocess.columns and 'Ram Mem' not in df_for_preprocess.columns:
        df_for_preprocess['Ram Mem'] = df_for_preprocess['RAM']
    
    # Prepare features
    df_processed, _ = preprocess_features(df_for_preprocess)
    feature_cols = get_feature_columns()
    
    # Identify quarter columns
    quarters = get_sorted_quarter_columns(df)
    if start_quarter in quarters:
        quarters = quarters[quarters.index(start_quarter):]
    
    # Find lifecycle for all products
    lifecycle = find_product_lifecycle(df, start_quarter)
    
    # Get all products that have at least one price (for prediction)
    predict_mask = df[quarters].notna().any(axis=1)
    all_products_to_predict = df.index[predict_mask]
    
    model_summary_rows = []
    coef_rows = []
    prediction_rows = []
    
    # Process each consecutive quarter pair
    for i in range(len(quarters) - 1):
        q1, q2 = quarters[i], quarters[i + 1]
        
        # Training: Get products that have prices in at least one of the two quarters
        mask_train = (df_processed[q1].notna() & (df_processed[q1] > 0)) | \
                    (df_processed[q2].notna() & (df_processed[q2] > 0))
        
        df_pair_train = df_processed.loc[mask_train].copy()
        if len(df_pair_train) < 10:
            continue
        
        # Pool data: create rows for each product-quarter combination
        pooled_data = []
        pooled_targets = []
        time_dummy = []
        
        for idx in df_pair_train.index:
            # Add row for quarter 1 if price exists
            if pd.notna(df_pair_train.loc[idx, q1]) and df_pair_train.loc[idx, q1] > 0:
                row_data = df_pair_train.loc[idx, feature_cols].values.copy()
                pooled_data.append(row_data)
                pooled_targets.append(np.log(df_pair_train.loc[idx, q1]))
                time_dummy.append(0)  # Quarter 1
            
            # Add row for quarter 2 if price exists
            if pd.notna(df_pair_train.loc[idx, q2]) and df_pair_train.loc[idx, q2] > 0:
                row_data = df_pair_train.loc[idx, feature_cols].values.copy()
                pooled_data.append(row_data)
                pooled_targets.append(np.log(df_pair_train.loc[idx, q2]))
                time_dummy.append(1)  # Quarter 2
        
        if len(pooled_data) < 10:
            continue
        
        X_pooled = np.array(pooled_data, dtype=float)
        y_pooled = np.array(pooled_targets, dtype=float)
        time_dummy = np.array(time_dummy, dtype=float)
        
        # Add time dummy as a feature
        X_with_time = np.column_stack([X_pooled, time_dummy])
        
        # Fill missing values
        if np.isnan(X_with_time).any():
            col_means = np.nanmean(X_with_time, axis=0)
            nan_mask = np.isnan(X_with_time)
            X_with_time[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_with_time)
        
        # Train OLS (LinearRegression) model
        ols = LinearRegression()
        ols.fit(X_scaled, y_pooled)
        
        r2 = ols.score(X_scaled, y_pooled)
        alpha = np.nan  # No regularization parameter for OLS
        n_features = int(len(ols.coef_))  # All coefficients are used
        
        # Prediction: Include all products that should be predicted for this quarter pair
        products_to_predict = []
        for product_idx in all_products_to_predict:
            if product_idx in lifecycle:
                life = lifecycle[product_idx]
                q1_idx = quarters.index(q1)
                q2_idx = quarters.index(q2)
                # Check if this quarter pair is within the product's lifecycle range
                if life['start_idx'] <= q1_idx and q2_idx <= life['end_idx']:
                    products_to_predict.append(product_idx)
        
        if len(products_to_predict) == 0:
            continue
        
        # Predict for all products in lifecycle range
        df_pair_predict = df_processed.loc[products_to_predict].copy()
        
        # Prepare features for prediction
        X_predict_base = df_pair_predict[feature_cols].fillna(df_processed[feature_cols].mean()).values
        
        # Predict for quarter 1 (time_dummy = 0)
        time_dummy_q1 = np.zeros(len(products_to_predict))
        X_predict_q1 = np.column_stack([X_predict_base, time_dummy_q1])
        X_predict_q1_scaled = scaler.transform(X_predict_q1)
        log_prices_pred_q1 = ols.predict(X_predict_q1_scaled)
        
        # Predict for quarter 2 (time_dummy = 1)
        time_dummy_q2 = np.ones(len(products_to_predict))
        X_predict_q2 = np.column_stack([X_predict_base, time_dummy_q2])
        X_predict_q2_scaled = scaler.transform(X_predict_q2)
        log_prices_pred_q2 = ols.predict(X_predict_q2_scaled)
        
        # Calculate actual prices for products that have them
        log_prices_actual_q1 = []
        log_prices_actual_q2 = []
        has_actual_q1 = []
        has_actual_q2 = []
        
        for product_idx in products_to_predict:
            if pd.notna(df_processed.loc[product_idx, q1]) and df_processed.loc[product_idx, q1] > 0:
                log_prices_actual_q1.append(np.log(df_processed.loc[product_idx, q1]))
                has_actual_q1.append(True)
            else:
                log_prices_actual_q1.append(np.nan)
                has_actual_q1.append(False)
            
            if pd.notna(df_processed.loc[product_idx, q2]) and df_processed.loc[product_idx, q2] > 0:
                log_prices_actual_q2.append(np.log(df_processed.loc[product_idx, q2]))
                has_actual_q2.append(True)
            else:
                log_prices_actual_q2.append(np.nan)
                has_actual_q2.append(False)
        
        log_prices_actual_q1 = np.array(log_prices_actual_q1)
        log_prices_actual_q2 = np.array(log_prices_actual_q2)
        
        # Calculate Jevons indices
        # Note: For Traditional, we should use mean(log_delta) not mean(log_Q2) - mean(log_Q1)
        # But we calculate deltas per product, so we'll use the deltas in calculate_jevons_indices
        # Here we calculate a summary statistic (mean of means) for model summary only
        # The actual Traditional Jevons Index is calculated correctly in calculate_jevons_indices()
        # using mean(Log_Delta_Actual) which equals mean(log(price_Q2) - log(price_Q1))
        log_deltas_actual_array = log_prices_actual_q2 - log_prices_actual_q1
        jevons_actual = float(np.nanmean(log_deltas_actual_array)) if (np.any(has_actual_q1) and np.any(has_actual_q2)) else np.nan
        log_deltas_predicted_array = log_prices_pred_q2 - log_prices_pred_q1
        jevons_predicted = float(np.nanmean(log_deltas_predicted_array))
        
        model_summary_rows.append({
            'Quarter_1': q1,
            'Quarter_2': q2,
            'Samples_Training': len(df_pair_train),
            'Samples_Pooled': len(pooled_data),
            'Samples_Prediction': len(products_to_predict),
            'R2_Score': r2,
            'Alpha': alpha,
            'Features_Selected': n_features,
            'Jevons_Actual': jevons_actual,
            'Jevons_Predicted': jevons_predicted,
            'Delta_Actual_%': jevons_actual * 100.0 if not np.isnan(jevons_actual) else np.nan,
            'Delta_Predicted_%': jevons_predicted * 100.0,
        })
        
        # Store coefficients
        for j, f in enumerate(feature_cols):
            coef_rows.append({
                'Quarter_1': q1,
                'Quarter_2': q2,
                'Feature': f,
                'Coefficient': ols.coef_[j],
                'Abs_Coefficient': abs(ols.coef_[j])
            })
        
        # Store time dummy coefficient
        coef_rows.append({
            'Quarter_1': q1,
            'Quarter_2': q2,
            'Feature': 'time_dummy',
            'Coefficient': ols.coef_[-1],
            'Abs_Coefficient': abs(ols.coef_[-1])
        })
        
        # Save predictions
        id_cols = ['Company Name', 'Model Name']
        asin_cols = [col for col in df_pair_predict.columns if 'ASIN' in col]
        id_cols = [col for col in id_cols if col in df_pair_predict.columns] + asin_cols
        
        tmp = df_pair_predict[id_cols].copy()
        tmp['Quarter_1'] = q1
        tmp['Quarter_2'] = q2
        tmp['Log_Price_Actual_Q1'] = log_prices_actual_q1
        tmp['Log_Price_Predicted_Q1'] = log_prices_pred_q1
        tmp['Price_Predicted_Q1'] = np.exp(log_prices_pred_q1)
        tmp['Price_Actual_Q1'] = np.exp(log_prices_actual_q1)
        tmp['Has_Actual_Q1'] = has_actual_q1
        
        tmp['Log_Price_Actual_Q2'] = log_prices_actual_q2
        tmp['Log_Price_Predicted_Q2'] = log_prices_pred_q2
        tmp['Price_Predicted_Q2'] = np.exp(log_prices_pred_q2)
        tmp['Price_Actual_Q2'] = np.exp(log_prices_actual_q2)
        tmp['Has_Actual_Q2'] = has_actual_q2
        
        tmp['Log_Delta_Actual'] = log_prices_actual_q2 - log_prices_actual_q1
        tmp['Log_Delta_Predicted'] = log_prices_pred_q2 - log_prices_pred_q1
        
        prediction_rows.append(tmp)
    
    model_df = pd.DataFrame(model_summary_rows)
    coef_df = pd.DataFrame(coef_rows)
    prediction_df = pd.concat(prediction_rows, axis=0) if prediction_rows else pd.DataFrame()
    
    return model_df, coef_df, prediction_df, df_processed, quarters


def calculate_jevons_indices(prediction_df):
    """
    Calculate Traditional and Hedonic Jevons indices from predictions
    """
    results = []
    
    # Group by quarter pairs
    quarter_pairs = prediction_df.groupby(['Quarter_1', 'Quarter_2'])
    
    for (q1, q2), group in quarter_pairs:
        # Traditional: mean of actual log deltas
        log_deltas_actual = group['Log_Delta_Actual'].dropna()
        if len(log_deltas_actual) > 0:
            mean_log_delta_traditional = float(np.nanmean(log_deltas_actual))
        else:
            mean_log_delta_traditional = np.nan
        
        # Hedonic: mean of predicted log deltas
        log_deltas_predicted = group['Log_Delta_Predicted'].dropna()
        if len(log_deltas_predicted) > 0:
            mean_log_delta_hedonic = float(np.nanmean(log_deltas_predicted))
        else:
            mean_log_delta_hedonic = np.nan
        
        results.append({
            'Quarter_1': q1,
            'Quarter_2': q2,
            'Period': f"{q1} → {q2}",
            'Mean_Log_Delta_Traditional': mean_log_delta_traditional,
            'Mean_Log_Delta_Hedonic': mean_log_delta_hedonic,
            'Price_Change_%_Traditional': mean_log_delta_traditional * 100.0 if not np.isnan(mean_log_delta_traditional) else np.nan,
            'Price_Change_%_Hedonic': mean_log_delta_hedonic * 100.0 if not np.isnan(mean_log_delta_hedonic) else np.nan,
            'Difference': mean_log_delta_hedonic - mean_log_delta_traditional if (not np.isnan(mean_log_delta_hedonic) and not np.isnan(mean_log_delta_traditional)) else np.nan,
            'Difference_%': (mean_log_delta_hedonic - mean_log_delta_traditional) * 100.0 if (not np.isnan(mean_log_delta_hedonic) and not np.isnan(mean_log_delta_traditional)) else np.nan,
            'Number_of_Products': len(group)
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate cumulative indices
    if not results_df.empty:
        cum_traditional = results_df['Mean_Log_Delta_Traditional'].sum()
        cum_hedonic = results_df['Mean_Log_Delta_Hedonic'].sum()
        cum_diff = cum_hedonic - cum_traditional
        
        print(f'\n=== Cumulative Comparison ===')
        print(f'Traditional (Actual): {cum_traditional:.6f} ({cum_traditional * 100:.2f}%)')
        print(f'Hedonic (Quality-Adjusted): {cum_hedonic:.6f} ({cum_hedonic * 100:.2f}%)')
        print(f'Difference: {cum_diff:.6f} ({cum_diff * 100:.2f}%)')
        if cum_traditional != 0:
            print(f'Quality Adjustment Effect: {abs(cum_diff) / abs(cum_traditional) * 100:.2f}%')
    
    return results_df


def create_model_period_matrix(prediction_df: pd.DataFrame, df_processed: pd.DataFrame, quarters: List[str]):
    """
    Create a matrix with Model Name (or Company Name + Model Name) as columns and period pairs as rows
    Each cell contains log_delta_predicted for that model in that period
    Only fills values for periods within each product's lifecycle (already filtered in prediction_df)
    
    Args:
        prediction_df: DataFrame with Log_Delta_Predicted, Company Name, Model Name, Quarter_1, Quarter_2
        df_processed: Original processed DataFrame with product information
        quarters: List of all quarter column names
    
    Returns:
        DataFrame with period pairs as index and Model identifiers as columns
    """
    if prediction_df.empty:
        print('Warning: prediction_df is empty, cannot create matrix')
        return pd.DataFrame()
    
    # Check required columns
    required_cols = ['Quarter_1', 'Quarter_2', 'Log_Delta_Predicted']
    missing_cols = [col for col in required_cols if col not in prediction_df.columns]
    if missing_cols:
        print(f'Error: Missing columns in prediction_df: {missing_cols}')
        print(f'Available columns: {prediction_df.columns.tolist()}')
        return pd.DataFrame()
    
    # Create model identifier: use Model Name if available, otherwise Company Name + Model Name
    if 'Model Name' in prediction_df.columns:
        if 'Company Name' in prediction_df.columns:
            # Use Company Name + Model Name for uniqueness
            prediction_df = prediction_df.copy()
            prediction_df['Model_Identifier'] = prediction_df['Company Name'].astype(str) + ' - ' + prediction_df['Model Name'].astype(str)
        else:
            prediction_df = prediction_df.copy()
            prediction_df['Model_Identifier'] = prediction_df['Model Name'].astype(str)
    elif 'Company Name' in prediction_df.columns:
        prediction_df = prediction_df.copy()
        prediction_df['Model_Identifier'] = prediction_df['Company Name'].astype(str)
    else:
        print('Error: Neither Model Name nor Company Name found in prediction_df')
        return pd.DataFrame()
    
    # Get all unique period pairs (sorted)
    period_pairs = []
    for i in range(len(quarters) - 1):
        q_prev, q_curr = quarters[i], quarters[i + 1]
        period_pairs.append(f"{q_prev}→{q_curr}")
    
    # Get all unique model identifiers from prediction_df
    model_identifiers = sorted(prediction_df['Model_Identifier'].dropna().unique())
    
    if len(model_identifiers) == 0:
        print('Warning: No model identifiers found in prediction_df')
        return pd.DataFrame()
    
    print(f'Found {len(model_identifiers)} models and {len(period_pairs)} period pairs')
    
    # Create empty matrix with all period pairs and all models
    matrix_data = {}
    for model_id in model_identifiers:
        matrix_data[model_id] = [np.nan] * len(period_pairs)
    
    # Fill matrix with log_delta_predicted values (one row per model-period combination)
    filled_count = 0
    for _, row in prediction_df.iterrows():
        model_id = row['Model_Identifier']
        q1 = row['Quarter_1']
        q2 = row['Quarter_2']
        log_delta = row['Log_Delta_Predicted']
        
        # Find the period pair index (using Quarter_1 and Quarter_2)
        period_str = f"{q1}→{q2}"
        if period_str in period_pairs:
            period_idx = period_pairs.index(period_str)
            if model_id in matrix_data:
                # If multiple entries for same model-period (shouldn't happen), keep the last one
                matrix_data[model_id][period_idx] = log_delta
                filled_count += 1
    
    print(f'Filled {filled_count} cells in the matrix')
    
    # Create DataFrame with period pairs as index
    matrix_df = pd.DataFrame(matrix_data, index=period_pairs)
    
    return matrix_df


def main():
    print('Reading Dataset.xlsx...')
    df = pd.read_excel('../Dataset.xlsx')
    
    print('Running time dummy models...')
    model_df, coef_df, prediction_df, df_processed, quarters = run_time_dummy_models(df, start_quarter='2020 Q1')
    
    print('Calculating Traditional and Hedonic Jevons Indices...')
    jevons_df = calculate_jevons_indices(prediction_df)

    print('Creating model-period matrix...')
    try:
        model_period_matrix = create_model_period_matrix(prediction_df, df_processed, quarters)
        if model_period_matrix.empty:
            print('Warning: Model-period matrix is empty!')
        else:
            print(f'Model-period matrix created: {model_period_matrix.shape[0]} periods × {model_period_matrix.shape[1]} models')
    except Exception as e:
        print(f'Error creating model-period matrix: {e}')
        import traceback
        traceback.print_exc()
        model_period_matrix = pd.DataFrame()
    
    out = 'Lasso_Time_Dummy_Models.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        model_df.to_excel(writer, sheet_name='Model_Summary', index=False)
        coef_df.to_excel(writer, sheet_name='Coefficients', index=False)
        if not prediction_df.empty:
            prediction_df.to_excel(writer, sheet_name='Predictions_By_Product', index=False)
        jevons_df.to_excel(writer, sheet_name='Jevons_Indices', index=False)
        # Always write the matrix, even if empty, so we can debug
        model_period_matrix.to_excel(writer, sheet_name='Model_Period_Matrix', index=True)
    
    print(f'\nWrote {out}')
    print('\n=== Time Dummy Model Summary ===')
    if not model_df.empty:
        print(model_df.head(10).to_string(index=False))
    
    print('\n=== Traditional vs Hedonic Jevons Index ===')
    if not jevons_df.empty:
        print(jevons_df[['Period', 'Mean_Log_Delta_Traditional', 'Mean_Log_Delta_Hedonic', 
                        'Difference', 'Difference_%']].to_string(index=False))


if __name__ == '__main__':
    main()

