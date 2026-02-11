import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List
import sys
import os
import statsmodels.api as sm

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
    Returns lifecycle dict with start and end years for prediction
    """
    # First aggregate quarterly data to annual
    annual_df = aggregate_quarters_to_years(df)
    years = get_sorted_year_columns(annual_df)
    if start_year in years:
        years = years[years.index(start_year):]
    
    lifecycle = {}
    
    for idx, row in annual_df.iterrows():
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
            # Start from one year before entry
            start_idx = max(0, entry_idx - 1)
            start_year_for_product = years[start_idx]
            
            # End at exit year + 1, or until last available year (2025)
            exit_idx = years.index(exit_year)
            end_idx = min(len(years) - 1, exit_idx + 1)
            end_year_for_product = years[end_idx]
            
            lifecycle[idx] = {
                'entry_year': entry_year,
                'exit_year': exit_year,
                'start_year': start_year_for_product,  # One year before entry
                'end_year': end_year_for_product,  # One year after exit (or until 2025)
                'start_idx': start_idx,
                'end_idx': end_idx,
                'years_to_predict': years[start_idx:end_idx+1]
            }
    
    return lifecycle


def run_time_dummy_models_annual(df, start_year='2020'):
    """
    Run time dummy models for annual data: pool consecutive years together, add time dummy variable,
    train a single Lasso model, and predict prices for both years
    """
    # First aggregate quarterly data to annual
    print("Aggregating quarterly data to annual averages...")
    annual_df = aggregate_quarters_to_years(df)
    
    # Map column names to match what preprocess_features expects
    df_for_preprocess = annual_df.copy()
    if 'RAM' in df_for_preprocess.columns and 'Ram Mem' not in df_for_preprocess.columns:
        df_for_preprocess['Ram Mem'] = df_for_preprocess['RAM']
    
    # Prepare features
    df_processed, _ = preprocess_features(df_for_preprocess)
    feature_cols = get_feature_columns()
    
    # Identify year columns
    years = get_sorted_year_columns(df_processed)
    if start_year in years:
        years = years[years.index(start_year):]
    
    # Find lifecycle for all products
    lifecycle = find_product_lifecycle(df, start_year)
    
    # Get all products that have at least one price (for prediction)
    predict_mask = annual_df[years].notna().any(axis=1)
    all_products_to_predict = annual_df.index[predict_mask]
    
    model_summary_rows = []
    coef_rows = []
    prediction_rows = []
    
    # Process each consecutive year pair
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        
        # Training: Get products that have prices in at least one of the two years
        mask_train = (df_processed[y1].notna() & (df_processed[y1] > 0)) | \
                    (df_processed[y2].notna() & (df_processed[y2] > 0))
        
        df_pair_train = df_processed.loc[mask_train].copy()
        if len(df_pair_train) < 10:
            continue
        
        # Pool data: create rows for each product-year combination
        pooled_data = []
        pooled_targets = []
        time_dummy = []
        
        for idx in df_pair_train.index:
            # Add row for year 1 if price exists
            if pd.notna(df_pair_train.loc[idx, y1]) and df_pair_train.loc[idx, y1] > 0:
                row_data = df_pair_train.loc[idx, feature_cols].values.copy()
                pooled_data.append(row_data)
                pooled_targets.append(np.log(df_pair_train.loc[idx, y1]))
                time_dummy.append(0)  # Year 1
            
            # Add row for year 2 if price exists
            if pd.notna(df_pair_train.loc[idx, y2]) and df_pair_train.loc[idx, y2] > 0:
                row_data = df_pair_train.loc[idx, feature_cols].values.copy()
                pooled_data.append(row_data)
                pooled_targets.append(np.log(df_pair_train.loc[idx, y2]))
                time_dummy.append(1)  # Year 2
        
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
        
        # Prediction: Include all products that should be predicted for this year pair
        products_to_predict = []
        for product_idx in all_products_to_predict:
            if product_idx in lifecycle:
                life = lifecycle[product_idx]
                y1_idx = years.index(y1)
                y2_idx = years.index(y2)
                # Check if this year pair is within the product's lifecycle range
                if life['start_idx'] <= y1_idx and y2_idx <= life['end_idx']:
                    products_to_predict.append(product_idx)
        
        if len(products_to_predict) == 0:
            continue
        
        # Predict for all products in lifecycle range
        df_pair_predict = df_processed.loc[products_to_predict].copy()
        
        # Prepare features for prediction
        X_predict_base = df_pair_predict[feature_cols].fillna(df_processed[feature_cols].mean()).values
        
        # Predict for year 1 (time_dummy = 0)
        time_dummy_y1 = np.zeros(len(products_to_predict))
        X_predict_y1 = np.column_stack([X_predict_base, time_dummy_y1])
        X_predict_y1_scaled = scaler.transform(X_predict_y1)
        log_prices_pred_y1 = ols.predict(X_predict_y1_scaled)
        
        # Predict for year 2 (time_dummy = 1)
        time_dummy_y2 = np.ones(len(products_to_predict))
        X_predict_y2 = np.column_stack([X_predict_base, time_dummy_y2])
        X_predict_y2_scaled = scaler.transform(X_predict_y2)
        log_prices_pred_y2 = ols.predict(X_predict_y2_scaled)
        
        # Calculate actual prices for products that have them
        log_prices_actual_y1 = []
        log_prices_actual_y2 = []
        has_actual_y1 = []
        has_actual_y2 = []
        
        for product_idx in products_to_predict:
            if pd.notna(df_processed.loc[product_idx, y1]) and df_processed.loc[product_idx, y1] > 0:
                log_prices_actual_y1.append(np.log(df_processed.loc[product_idx, y1]))
                has_actual_y1.append(True)
            else:
                log_prices_actual_y1.append(np.nan)
                has_actual_y1.append(False)
            
            if pd.notna(df_processed.loc[product_idx, y2]) and df_processed.loc[product_idx, y2] > 0:
                log_prices_actual_y2.append(np.log(df_processed.loc[product_idx, y2]))
                has_actual_y2.append(True)
            else:
                log_prices_actual_y2.append(np.nan)
                has_actual_y2.append(False)
        
        log_prices_actual_y1 = np.array(log_prices_actual_y1)
        log_prices_actual_y2 = np.array(log_prices_actual_y2)
        
        # Calculate Jevons indices
        # Note: For Traditional, we should use mean(log_delta) not mean(log_Y2) - mean(log_Y1)
        # But we calculate deltas per product, so we'll use the deltas in calculate_jevons_indices
        # Here we calculate a summary statistic (mean of means) for model summary only
        # The actual Traditional Jevons Index is calculated correctly in calculate_jevons_indices()
        # using mean(Log_Delta_Actual) which equals mean(log(price_Y2) - log(price_Y1))
        log_deltas_actual_array = log_prices_actual_y2 - log_prices_actual_y1
        jevons_actual = float(np.nanmean(log_deltas_actual_array)) if (np.any(has_actual_y1) and np.any(has_actual_y2)) else np.nan
        log_deltas_predicted_array = log_prices_pred_y2 - log_prices_pred_y1
        jevons_predicted = float(np.nanmean(log_deltas_predicted_array))
        
        model_summary_rows.append({
            'Year_1': y1,
            'Year_2': y2,
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
                'Year_1': y1,
                'Year_2': y2,
                'Feature': f,
                'Coefficient': ols.coef_[j],
                'Abs_Coefficient': abs(ols.coef_[j])
            })
        
        # Store time dummy coefficient
        coef_rows.append({
            'Year_1': y1,
            'Year_2': y2,
            'Feature': 'time_dummy',
            'Coefficient': ols.coef_[-1],
            'Abs_Coefficient': abs(ols.coef_[-1])
        })
        
        # Save predictions
        id_cols = ['Company Name', 'Model Name']
        asin_cols = [col for col in df_pair_predict.columns if 'ASIN' in col]
        id_cols = [col for col in id_cols if col in df_pair_predict.columns] + asin_cols
        
        tmp = df_pair_predict[id_cols].copy()
        tmp['Year_1'] = y1
        tmp['Year_2'] = y2
        tmp['Log_Price_Actual_Y1'] = log_prices_actual_y1
        tmp['Log_Price_Predicted_Y1'] = log_prices_pred_y1
        tmp['Price_Predicted_Y1'] = np.exp(log_prices_pred_y1)
        tmp['Price_Actual_Y1'] = np.exp(log_prices_actual_y1)
        tmp['Has_Actual_Y1'] = has_actual_y1
        
        tmp['Log_Price_Actual_Y2'] = log_prices_actual_y2
        tmp['Log_Price_Predicted_Y2'] = log_prices_pred_y2
        tmp['Price_Predicted_Y2'] = np.exp(log_prices_pred_y2)
        tmp['Price_Actual_Y2'] = np.exp(log_prices_actual_y2)
        tmp['Has_Actual_Y2'] = has_actual_y2
        
        tmp['Log_Delta_Actual'] = log_prices_actual_y2 - log_prices_actual_y1
        tmp['Log_Delta_Predicted'] = log_prices_pred_y2 - log_prices_pred_y1
        
        prediction_rows.append(tmp)
    
    model_df = pd.DataFrame(model_summary_rows)
    coef_df = pd.DataFrame(coef_rows)
    prediction_df = pd.concat(prediction_rows, axis=0) if prediction_rows else pd.DataFrame()
    
    return model_df, coef_df, prediction_df, df_processed, years


def calculate_jevons_indices(prediction_df):
    """
    Calculate Traditional and Hedonic Jevons indices from predictions
    """
    results = []
    
    # Group by year pairs
    year_pairs = prediction_df.groupby(['Year_1', 'Year_2'])
    
    for (y1, y2), group in year_pairs:
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
            'Year_1': y1,
            'Year_2': y2,
            'Period': f"{y1} → {y2}",
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


def calculate_price_change_r2_from_deltas(prediction_df):
    """
    Calculate price-change R² for each year pair by regressing
        Log_Delta_Actual on Log_Delta_Predicted
    using the annual time-dummy model's delta outputs.
    
    Note: For time-dummy models, if product features don't change between years,
    Log_Delta_Predicted will be constant (equal to time dummy coefficient) for all products,
    leading to R² = 0. This is expected behavior, not a bug.
    """
    if prediction_df.empty:
        return {}

    interval_r2 = {}

    year_pairs = prediction_df.groupby(['Year_1', 'Year_2'])
    for (y1, y2), group in year_pairs:
        g = group.dropna(subset=['Log_Delta_Actual', 'Log_Delta_Predicted'])
        if len(g) < 5:
            continue

        y = g['Log_Delta_Actual'].values
        x = g['Log_Delta_Predicted'].values
        
        # Check variance of predicted deltas
        x_var = np.var(x)
        x_mean = np.mean(x)
        x_std = np.std(x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        # If predicted deltas have zero or very small variance, R² will be 0
        # This is expected for time-dummy models when product features don't change
        if x_var < 1e-10:
            print(f"{y1} → {y2}: Log_Delta_Predicted has zero variance (all values ≈ {x_mean:.6f}), "
                  f"R² = 0 (expected for time-dummy when features constant) (n={len(g)})")
            r2 = 0.0
        else:
            X = sm.add_constant(x)
            ols = sm.OLS(y, X).fit()
            r2 = float(ols.rsquared)
            print(f"{y1} → {y2}: R² (price-change, regression, annual time-dummy) = {r2:.4f} (n={len(g)}, "
                  f"pred_delta: mean={x_mean:.4f}, std={x_std:.4f}, "
                  f"actual_delta: mean={y_mean:.4f}, std={y_std:.4f})")

        label = f"{y1} → {y2}"
        interval_r2[label] = {
            'r2': r2,
            'n_samples': len(g)
        }

    if interval_r2:
        avg_r2 = np.mean([v['r2'] for v in interval_r2.values()])
        print(f"\nOverall Price-Change R² (annual time-dummy, average across years): {avg_r2:.4f}")

    return interval_r2


def create_model_period_matrix(prediction_df: pd.DataFrame, df_processed: pd.DataFrame, years: List[str]):
    """
    Create a matrix with Model Name (or Company Name + Model Name) as columns and year pairs as rows
    Each cell contains log_delta_predicted for that model in that period
    Only fills values for periods within each product's lifecycle (already filtered in prediction_df)
    
    Args:
        prediction_df: DataFrame with Log_Delta_Predicted, Company Name, Model Name, Year_1, Year_2
        df_processed: Original processed DataFrame with product information
        years: List of all year column names (as strings)
    
    Returns:
        DataFrame with year pairs as index and Model identifiers as columns
    """
    if prediction_df.empty:
        print('Warning: prediction_df is empty, cannot create matrix')
        return pd.DataFrame()
    
    # Check required columns
    required_cols = ['Year_1', 'Year_2', 'Log_Delta_Predicted']
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
    
    # Get all unique year pairs (sorted)
    year_pairs = []
    for i in range(len(years) - 1):
        year_prev, year_curr = years[i], years[i + 1]
        year_pairs.append(f"{year_prev}→{year_curr}")
    
    # Get all unique model identifiers, preserving original dataset order
    # Create model identifier from df_processed to preserve original order
    df_processed_copy = None
    if 'Model Name' in df_processed.columns:
        if 'Company Name' in df_processed.columns:
            df_processed_copy = df_processed.copy()
            df_processed_copy['Model_Identifier'] = df_processed_copy['Company Name'].astype(str) + ' - ' + df_processed_copy['Model Name'].astype(str)
        else:
            df_processed_copy = df_processed.copy()
            df_processed_copy['Model_Identifier'] = df_processed_copy['Model Name'].astype(str)
    elif 'Company Name' in df_processed.columns:
        df_processed_copy = df_processed.copy()
        df_processed_copy['Model_Identifier'] = df_processed_copy['Company Name'].astype(str)
    
    # Use drop_duplicates() to preserve first occurrence order from df_processed
    # If df_processed has the columns, use it; otherwise fallback to prediction_df
    if df_processed_copy is not None:
        model_identifiers = df_processed_copy['Model_Identifier'].dropna().drop_duplicates().tolist()
    else:
        # Fallback to prediction_df if df_processed doesn't have the columns
        model_identifiers = prediction_df['Model_Identifier'].dropna().drop_duplicates().tolist()
    
    if len(model_identifiers) == 0:
        print('Warning: No model identifiers found')
        return pd.DataFrame()
    
    print(f'Found {len(model_identifiers)} models and {len(year_pairs)} year pairs')
    
    # Create empty matrix with all year pairs and all models
    matrix_data = {}
    for model_id in model_identifiers:
        matrix_data[model_id] = [np.nan] * len(year_pairs)
    
    # Fill matrix with log_delta_predicted values (one row per model-period combination)
    filled_count = 0
    for _, row in prediction_df.iterrows():
        model_id = row['Model_Identifier']
        y1 = str(int(row['Year_1'])) if pd.notna(row['Year_1']) else None
        y2 = str(int(row['Year_2'])) if pd.notna(row['Year_2']) else None
        
        if y1 is None or y2 is None:
            continue
        
        log_delta = row['Log_Delta_Predicted']
        
        # Find the year pair index (using Year_1 and Year_2)
        year_str = f"{y1}→{y2}"
        if year_str in year_pairs:
            year_idx = year_pairs.index(year_str)
            if model_id in matrix_data:
                # If multiple entries for same model-period (shouldn't happen), keep the last one
                matrix_data[model_id][year_idx] = log_delta
                filled_count += 1
    
    print(f'Filled {filled_count} cells in the matrix')
    
    # Create DataFrame with year pairs as index
    matrix_df = pd.DataFrame(matrix_data, index=year_pairs)
    
    return matrix_df


def main():
    print('Reading Dataset.xlsx...')
    df = pd.read_excel('../Dataset.xlsx')
    
    print('Running annual time dummy models...')
    model_df, coef_df, prediction_df, df_processed, years = run_time_dummy_models_annual(df, start_year='2020')
    
    print('Calculating Traditional and Hedonic Jevons Indices...')
    jevons_df = calculate_jevons_indices(prediction_df)

    print('\nCalculating Price-Change R² via Regression (Annual Time-Dummy Model)...')
    interval_r2 = calculate_price_change_r2_from_deltas(prediction_df)

    print('Creating model-period matrix...')
    try:
        model_period_matrix = create_model_period_matrix(prediction_df, df_processed, years)
        if model_period_matrix.empty:
            print('Warning: Model-period matrix is empty!')
        else:
            print(f'Model-period matrix created: {model_period_matrix.shape[0]} periods × {model_period_matrix.shape[1]} models')
    except Exception as e:
        print(f'Error creating model-period matrix: {e}')
        import traceback
        traceback.print_exc()
        model_period_matrix = pd.DataFrame()
    
    out = 'Lasso_Time_Dummy_Models_Annual.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        model_df.to_excel(writer, sheet_name='Model_Summary', index=False)
        coef_df.to_excel(writer, sheet_name='Coefficients', index=False)
        if not prediction_df.empty:
            prediction_df.to_excel(writer, sheet_name='Predictions_By_Product', index=False)
        jevons_df.to_excel(writer, sheet_name='Jevons_Indices', index=False)
        # Always write the matrix, even if empty, so we can debug
        model_period_matrix.to_excel(writer, sheet_name='Model_Period_Matrix', index=True)
        # Price-change R² by interval (if available)
        if interval_r2:
            r2_rows = []
            for period, info in interval_r2.items():
                r2_rows.append({
                    'Period': period,
                    'R2_Price_Change': info['r2'],
                    'Samples': info['n_samples']
                })
            r2_df = pd.DataFrame(r2_rows)
            avg_r2 = r2_df['R2_Price_Change'].mean()
            summary_row = pd.DataFrame({
                'Period': ['Overall (Average)'],
                'R2_Price_Change': [avg_r2],
                'Samples': [r2_df['Samples'].sum()]
            })
            r2_df = pd.concat([r2_df, summary_row], ignore_index=True)
            r2_df.to_excel(writer, sheet_name='Price_Change_R2', index=False)
    
    print(f'\nWrote {out}')
    print('\n=== Annual Time Dummy Model Summary ===')
    if not model_df.empty:
        print(model_df.head(10).to_string(index=False))
    
    print('\n=== Traditional vs Hedonic Jevons Index ===')
    if not jevons_df.empty:
        print(jevons_df[['Period', 'Mean_Log_Delta_Traditional', 'Mean_Log_Delta_Hedonic', 
                        'Difference', 'Difference_%']].to_string(index=False))


if __name__ == '__main__':
    main()

