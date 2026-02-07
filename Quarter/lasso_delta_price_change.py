import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import resample
import statsmodels.api as sm
from typing import List, Tuple

# Reuse feature preprocessing to keep definitions consistent
from lasso_price_prediction import preprocess_features, get_feature_columns


def parse_quarter(col_name: str) -> Tuple[int, int]:
    parts = str(col_name).split()
    if len(parts) >= 2 and 'Q' in parts[1]:
        return int(parts[0]), int(parts[1].replace('Q', ''))
    return None, None


def get_sorted_quarter_columns(df: pd.DataFrame) -> List[str]:
    quarter_columns = [c for c in df.columns if 'Q' in c and any(ch.isdigit() for ch in c)]
    quarter_columns = sorted(
        quarter_columns,
        key=lambda x: (int(x.split()[0]), int(x.split()[1][1:]))
    )
    return quarter_columns


def find_product_lifecycle_for_delta(df: pd.DataFrame, start_from: str = '2020 Q1'):
    """
    Find entry and exit quarters for each product
    Entry: first quarter with actual price
    Exit: last quarter with actual price
    Returns lifecycle dict with start and end quarters for prediction
    """
    quarters = get_sorted_quarter_columns(df)
    if start_from in quarters:
        quarters = quarters[quarters.index(start_from):]
    
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
            exit_idx = quarters.index(exit_quarter)
            
            # Start from one quarter before entry
            start_idx = max(0, entry_idx - 1)
            # End at one quarter after exit, or until last available quarter
            end_idx = min(len(quarters) - 1, exit_idx + 1)
            
            lifecycle[idx] = {
                'entry_quarter': entry_quarter,
                'exit_quarter': exit_quarter,
                'start_quarter': quarters[start_idx],
                'end_quarter': quarters[end_idx],
                'start_idx': start_idx,
                'end_idx': end_idx
            }
    
    return lifecycle


def run_delta_models(df: pd.DataFrame, start_from: str = '2020 Q1'):
    # Map column names to match what preprocess_features expects
    df_for_preprocess = df.copy()
    if 'RAM' in df_for_preprocess.columns and 'Ram Mem' not in df_for_preprocess.columns:
        df_for_preprocess['Ram Mem'] = df_for_preprocess['RAM']
    
    # Prepare features
    df_processed, _ = preprocess_features(df_for_preprocess)
    feature_cols = get_feature_columns()

    # Identify quarter columns
    quarters = get_sorted_quarter_columns(df)
    if start_from in quarters:
        quarters = quarters[quarters.index(start_from):]

    # Find lifecycle for all products
    lifecycle = find_product_lifecycle_for_delta(df, start_from)
    
    # Get all products that have at least one price (for prediction)
    predict_mask = df[quarters].notna().any(axis=1)
    all_products_to_predict = df.index[predict_mask]

    model_summary_rows = []
    coef_rows = []
    delta_rows = []
    
    # Store quarters for later use in matrix creation
    _quarters_for_matrix = quarters

    for i in range(len(quarters) - 1):
        q_prev, q_curr = quarters[i], quarters[i + 1]

        # Training: Require both quarters to have valid prices for the same product
        mask_train = df_processed[q_prev].notna() & (df_processed[q_prev] > 0) \
             & df_processed[q_curr].notna() & (df_processed[q_curr] > 0)

        df_pair_train = df_processed.loc[mask_train].copy()
        if len(df_pair_train) < 10:
            continue

        # Prediction: Include all products that should be predicted for this quarter pair
        # (i.e., products whose lifecycle includes this quarter pair)
        products_to_predict = []
        for product_idx in all_products_to_predict:
            if product_idx in lifecycle:
                life = lifecycle[product_idx]
                q_prev_idx = quarters.index(q_prev)
                q_curr_idx = quarters.index(q_curr)
                # Check if this quarter pair is within the product's lifecycle range
                if life['start_idx'] <= q_prev_idx and q_curr_idx <= life['end_idx']:
                    products_to_predict.append(product_idx)
        
        if len(products_to_predict) == 0:
            continue

        # Target: log price difference (for training, only products with actual prices)
        y = np.log(df_pair_train[q_curr]) - np.log(df_pair_train[q_prev])
        X = df_pair_train[feature_cols]

        # Fill missing features conservatively with column means (within the pair)
        if X.isnull().any().any():
            X = X.fillna(X.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LassoCV(cv=min(5, len(df_pair_train)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)

        r2 = lasso.score(X_scaled, y)
        alpha = lasso.alpha_
        n_features = int(np.sum(lasso.coef_ != 0))
        
        # Calculate bootstrap confidence intervals
        def bootstrap_lasso_coefficients(X_scaled, y, alpha, n_bootstraps=500, random_state=42):
            """Calculate bootstrap confidence intervals for Lasso coefficients"""
            np.random.seed(random_state)
            n_features = X_scaled.shape[1]
            lasso = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
            boot_coefs = []
            
            for _ in range(n_bootstraps):
                X_resampled, y_resampled = resample(X_scaled, y)
                lasso.fit(X_resampled, y_resampled)
                selected_features = np.where(lasso.coef_ != 0)[0]
                
                if len(selected_features) > 0:
                    X_selected = X_resampled[:, selected_features]
                    X_selected = sm.add_constant(X_selected)
                    ols = sm.OLS(y_resampled, X_selected).fit()
                    coef_full = np.zeros(n_features)
                    coef_full[selected_features] = ols.params[1:]
                    boot_coefs.append(coef_full)
                else:
                    boot_coefs.append(np.zeros(n_features))
            
            if len(boot_coefs) == 0:
                return None
            
            boot_coefs = np.array(boot_coefs)
            conf_intervals = np.percentile(boot_coefs, [2.5, 97.5], axis=0)
            return conf_intervals
        
        print(f"    Calculating bootstrap confidence intervals for {q_prev}->{q_curr} (n_bootstrap=500)...")
        conf_intervals = bootstrap_lasso_coefficients(
            X_scaled, y.values, alpha, n_bootstraps=500, random_state=42
        )

        # In-sample predictions (for training data only)
        y_hat_in = lasso.predict(X_scaled)
        # Out-of-fold predictions to avoid mean-matching artifact
        # Use more folds for better OOF separation; fallback to LOOCV when small
        k = min(10, max(2, len(df_pair_train)-1))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        y_hat_oof = np.empty_like(y.values)
        y_hat_oof[:] = np.nan
        test_r2 = []
        alphas = []
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            # fit scaler on train only
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            # inner CV to choose alpha on train
            lcv = LassoCV(cv=min(5, max(2, len(tr_idx)//2)), random_state=42, max_iter=2000)
            lcv.fit(X_tr_sc, y_tr)
            y_pred_te = lcv.predict(X_te_sc)
            # store
            y_hat_oof[te_idx] = y_pred_te
            # test R2
            ss_res = np.sum((y_te.values - y_pred_te)**2)
            ss_tot = np.sum((y_te.values - np.mean(y_te.values))**2)
            test_r2.append(1 - ss_res/ss_tot if ss_tot > 0 else np.nan)
            alphas.append(lcv.alpha_)

        jevons_actual = float(np.mean(y))
        # Always use OOF mean (nan-robust); do NOT fall back to in-sample
        jevons_pred = float(np.nanmean(y_hat_oof))
        r2_oof = float(np.nanmean(test_r2)) if test_r2 else np.nan
        alpha_oof = float(np.nanmean(alphas)) if alphas else np.nan

        # Predict for all products in lifecycle range (including those without actual prices)
        df_pair_predict = df_processed.loc[products_to_predict].copy()
        
        # Prepare features for prediction
        X_predict = df_pair_predict[feature_cols].fillna(df_processed[feature_cols].mean())
        X_predict_scaled = scaler.transform(X_predict)
        
        # Predict log deltas for all products
        log_deltas_predicted = lasso.predict(X_predict_scaled)
        
        # Calculate actual deltas for products that have both prices
        log_deltas_actual = []
        has_actual = []
        for product_idx in products_to_predict:
            if (pd.notna(df_processed.loc[product_idx, q_prev]) and df_processed.loc[product_idx, q_prev] > 0 and
                pd.notna(df_processed.loc[product_idx, q_curr]) and df_processed.loc[product_idx, q_curr] > 0):
                actual_delta = np.log(df_processed.loc[product_idx, q_curr]) - np.log(df_processed.loc[product_idx, q_prev])
                log_deltas_actual.append(actual_delta)
                has_actual.append(True)
            else:
                log_deltas_actual.append(np.nan)
                has_actual.append(False)
        
        log_deltas_actual = np.array(log_deltas_actual)

        model_summary_rows.append({
            'Base Quarter': q_prev,
            'Current Quarter': q_curr,
            'Samples_Training': len(df_pair_train),
            'Samples_Prediction': len(products_to_predict),
            'R2_Score': r2,
            'Alpha_InSample': alpha,
            'Features_Selected': n_features,
            'R2_OOF': r2_oof,
            'Alpha_OOF_Mean': alpha_oof,
            'Jevons_Actual': float(np.nanmean(log_deltas_actual)) if np.any(has_actual) else np.nan,
            'Jevons_Predicted': float(np.nanmean(log_deltas_predicted)),
            'Delta_Actual_%': (float(np.nanmean(log_deltas_actual)) - 1) * 100.0 if np.any(has_actual) else np.nan,
            'Delta_Predicted_%': (float(np.nanmean(log_deltas_predicted)) - 1) * 100.0,
        })

        for j, f in enumerate(feature_cols):
            coef_value = lasso.coef_[j]
            if conf_intervals is not None:
                lower_bound = conf_intervals[0, j]
                upper_bound = conf_intervals[1, j]
            else:
                lower_bound = np.nan
                upper_bound = np.nan
            
            coef_rows.append({
                'Base Quarter': q_prev,
                'Current Quarter': q_curr,
                'Feature': f,
                'Coefficient': coef_value,
                'Abs_Coefficient': abs(coef_value),
                'CI_Lower': lower_bound,
                'CI_Upper': upper_bound
            })

        # Save row-level deltas for all predicted products
        id_cols = ['Company Name', 'Model Name']
        asin_cols = [col for col in df_pair_predict.columns if 'ASIN' in col]
        id_cols = [col for col in id_cols if col in df_pair_predict.columns] + asin_cols
        tmp = df_pair_predict[id_cols].copy()
        tmp['Base Quarter'] = q_prev
        tmp['Current Quarter'] = q_curr
        tmp['Log_Delta_Actual'] = log_deltas_actual
        tmp['Log_Delta_Predicted'] = log_deltas_predicted
        tmp['Delta_Predicted'] = np.exp(log_deltas_predicted)
        tmp['Delta_Actual'] = np.exp(log_deltas_actual)
        tmp['Has_Actual_Prices'] = has_actual
        # Also store means with higher precision for diagnostics
        tmp['Mean_Log_Actual'] = float(np.nanmean(log_deltas_actual)) if np.any(has_actual) else np.nan
        tmp['Mean_Log_Pred'] = float(np.nanmean(log_deltas_predicted))
        delta_rows.append(tmp)

    model_df = pd.DataFrame(model_summary_rows)
    coef_df = pd.DataFrame(coef_rows)
    delta_df = pd.concat(delta_rows, axis=0) if delta_rows else pd.DataFrame()

    return model_df, coef_df, delta_df, df_processed, _quarters_for_matrix


def create_model_period_matrix(delta_df: pd.DataFrame, df_processed: pd.DataFrame, quarters: List[str]):
    """
    Create a matrix with Model Name (or Company Name + Model Name) as columns and period pairs as rows
    Each cell contains log_delta_predicted for that model in that period
    Only fills values for periods within each product's lifecycle (already filtered in delta_df)
    
    Args:
        delta_df: DataFrame with Log_Delta_Predicted, Company Name, Model Name, Base Quarter, Current Quarter
        df_processed: Original processed DataFrame with product information
        quarters: List of all quarter column names
    
    Returns:
        DataFrame with period pairs as index and Model identifiers as columns
    """
    if delta_df.empty:
        print('Warning: delta_df is empty, cannot create matrix')
        return pd.DataFrame()
    
    # Check required columns
    required_cols = ['Base Quarter', 'Current Quarter', 'Log_Delta_Predicted']
    missing_cols = [col for col in required_cols if col not in delta_df.columns]
    if missing_cols:
        print(f'Error: Missing columns in delta_df: {missing_cols}')
        print(f'Available columns: {delta_df.columns.tolist()}')
        return pd.DataFrame()
    
    # Create model identifier: use Model Name if available, otherwise Company Name + Model Name
    if 'Model Name' in delta_df.columns:
        if 'Company Name' in delta_df.columns:
            # Use Company Name + Model Name for uniqueness
            delta_df = delta_df.copy()
            delta_df['Model_Identifier'] = delta_df['Company Name'].astype(str) + ' - ' + delta_df['Model Name'].astype(str)
        else:
            delta_df = delta_df.copy()
            delta_df['Model_Identifier'] = delta_df['Model Name'].astype(str)
    elif 'Company Name' in delta_df.columns:
        delta_df = delta_df.copy()
        delta_df['Model_Identifier'] = delta_df['Company Name'].astype(str)
    else:
        print('Error: Neither Model Name nor Company Name found in delta_df')
        return pd.DataFrame()
    
    # Get all unique period pairs (sorted)
    period_pairs = []
    for i in range(len(quarters) - 1):
        q_prev, q_curr = quarters[i], quarters[i + 1]
        period_pairs.append(f"{q_prev}→{q_curr}")
    
    # Get all unique model identifiers from delta_df
    model_identifiers = sorted(delta_df['Model_Identifier'].dropna().unique())
    
    if len(model_identifiers) == 0:
        print('Warning: No model identifiers found in delta_df')
        return pd.DataFrame()
    
    print(f'Found {len(model_identifiers)} models and {len(period_pairs)} period pairs')
    
    # Create empty matrix with all period pairs and all models
    matrix_data = {}
    for model_id in model_identifiers:
        matrix_data[model_id] = [np.nan] * len(period_pairs)
    
    # Fill matrix with log_delta_predicted values (one row per model-period combination)
    filled_count = 0
    for _, row in delta_df.iterrows():
        model_id = row['Model_Identifier']
        base_q = row['Base Quarter']
        curr_q = row['Current Quarter']
        log_delta = row['Log_Delta_Predicted']
        
        # Find the period pair index
        period_str = f"{base_q}→{curr_q}"
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


def calculate_hedonic_jevons_from_deltas(delta_df: pd.DataFrame):
    """
    Calculate Hedonic Jevons Index directly from predicted price changes (deltas)
    Returns both hedonic (from predicted deltas) and traditional (from actual deltas) for comparison
    """
    hedonic_results = []
    traditional_results = []
    
    # Group by quarter pairs
    quarter_pairs = delta_df.groupby(['Base Quarter', 'Current Quarter'])
    
    for (q_prev, q_curr), group in quarter_pairs:
        # Get predicted and actual log deltas
        log_deltas_pred = group['Log_Delta_Predicted'].dropna()
        log_deltas_actual = group['Log_Delta_Actual'].dropna()
        
        if len(log_deltas_pred) > 0:
            # Hedonic: mean log price change from predicted deltas (quality-adjusted)
            mean_log_delta_hedonic = float(np.nanmean(log_deltas_pred))
            n_products = len(log_deltas_pred)
            
            hedonic_results.append({
                'Base Quarter': q_prev,
                'Current Quarter': q_curr,
                'Period': f"{q_prev} → {q_curr}",
                'Mean_Log_Delta_Hedonic': mean_log_delta_hedonic,
                'Number_of_Products': n_products,
                'Price_Change_%_Hedonic': mean_log_delta_hedonic * 100
            })
        
        if len(log_deltas_actual) > 0:
            # Traditional: mean log price change from actual deltas (no quality adjustment)
            mean_log_delta_traditional = float(np.nanmean(log_deltas_actual))
            
            traditional_results.append({
                'Base Quarter': q_prev,
                'Current Quarter': q_curr,
                'Period': f"{q_prev} → {q_curr}",
                'Mean_Log_Delta_Traditional': mean_log_delta_traditional,
                'Price_Change_%_Traditional': mean_log_delta_traditional * 100
            })
    
    hedonic_df = pd.DataFrame(hedonic_results)
    traditional_df = pd.DataFrame(traditional_results)
    
    # Merge for comparison
    comparison_df = hedonic_df.merge(traditional_df, on=['Base Quarter', 'Current Quarter', 'Period'], how='outer')
    comparison_df['Difference'] = comparison_df['Mean_Log_Delta_Hedonic'] - comparison_df['Mean_Log_Delta_Traditional']
    comparison_df['Difference_%'] = comparison_df['Difference'] * 100
    
    return hedonic_df, traditional_df, comparison_df


def main():
    print('Reading Dataset.xlsx...')
    df = pd.read_excel('../Dataset.xlsx')

    print('Running delta models...')
    model_df, coef_df, delta_df, df_processed, quarters = run_delta_models(df, start_from='2020 Q1')

    print('Calculating Traditional and Hedonic Jevons Indices for comparison...')
    hedonic_df, traditional_df, comparison_df = calculate_hedonic_jevons_from_deltas(delta_df)

    print('Creating model-period matrix...')
    try:
        model_period_matrix = create_model_period_matrix(delta_df, df_processed, quarters)
        if model_period_matrix.empty:
            print('Warning: Model-period matrix is empty!')
        else:
            print(f'Model-period matrix created: {model_period_matrix.shape[0]} periods × {model_period_matrix.shape[1]} models')
    except Exception as e:
        print(f'Error creating model-period matrix: {e}')
        import traceback
        traceback.print_exc()
        model_period_matrix = pd.DataFrame()

    out = 'Lasso_Delta_Models1.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        model_df.to_excel(writer, sheet_name='Model_Summary_Delta', index=False)
        coef_df.to_excel(writer, sheet_name='Coefficients_Delta', index=False)
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name='Deltas_By_Product', index=False)
        hedonic_df.to_excel(writer, sheet_name='Hedonic_Jevons', index=False)
        traditional_df.to_excel(writer, sheet_name='Traditional_Jevons', index=False)
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        # Always write the matrix, even if empty, so we can debug
        model_period_matrix.to_excel(writer, sheet_name='Model_Period_Matrix', index=True)

    print(f'\nWrote {out}')
    print('\n=== Delta Model Summary ===')
    if not model_df.empty:
        print(model_df.head(10).to_string(index=False))
    
    print('\n=== Comparison: Traditional vs Hedonic Jevons Index ===')
    if not comparison_df.empty:
        print(comparison_df[['Period', 'Mean_Log_Delta_Traditional', 'Mean_Log_Delta_Hedonic', 
                             'Difference', 'Difference_%']].to_string(index=False))
        
        cum_traditional = comparison_df['Mean_Log_Delta_Traditional'].sum()
        cum_hedonic = comparison_df['Mean_Log_Delta_Hedonic'].sum()
        cum_diff = cum_hedonic - cum_traditional
        
        print(f'\n=== Cumulative Comparison ===')
        print(f'Traditional (Actual): {cum_traditional:.6f} ({cum_traditional * 100:.2f}%)')
        print(f'Hedonic (Quality-Adjusted): {cum_hedonic:.6f} ({cum_hedonic * 100:.2f}%)')
        print(f'Difference: {cum_diff:.6f} ({cum_diff * 100:.2f}%)')
        print(f'Quality Adjustment Effect: {abs(cum_diff) / abs(cum_traditional) * 100 if cum_traditional != 0 else np.nan:.2f}%')


if __name__ == '__main__':
    main()


