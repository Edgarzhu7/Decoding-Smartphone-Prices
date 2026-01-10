import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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

        for f, c in zip(feature_cols, lasso.coef_):
            coef_rows.append({
                'Base Quarter': q_prev,
                'Current Quarter': q_curr,
                'Feature': f,
                'Coefficient': c,
                'Abs_Coefficient': abs(c)
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

    return model_df, coef_df, delta_df


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
    model_df, coef_df, delta_df = run_delta_models(df, start_from='2020 Q1')

    print('Calculating Traditional and Hedonic Jevons Indices for comparison...')
    hedonic_df, traditional_df, comparison_df = calculate_hedonic_jevons_from_deltas(delta_df)

    out = 'Lasso_Delta_Models1.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        model_df.to_excel(writer, sheet_name='Model_Summary_Delta', index=False)
        coef_df.to_excel(writer, sheet_name='Coefficients_Delta', index=False)
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name='Deltas_By_Product', index=False)
        hedonic_df.to_excel(writer, sheet_name='Hedonic_Jevons', index=False)
        traditional_df.to_excel(writer, sheet_name='Traditional_Jevons', index=False)
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

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


