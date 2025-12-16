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


def run_delta_models_with_error(df: pd.DataFrame, start_from: str = '2020 Q1'):
    """
    Run delta models with error feature from previous period pair
    Sequentially trains models: for period pair (t, t+1), uses error from pair (t-1, t) as feature
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
    if start_from in quarters:
        quarters = quarters[quarters.index(start_from):]

    model_summary_rows = []
    coef_rows = []
    delta_rows = []
    
    # Store errors from previous period pairs: {quarter_pair: {product_idx: error}}
    errors = {}  # Key: (q_prev, q_curr), Value: {product_idx: error}

    for i in range(len(quarters) - 1):
        q_prev, q_curr = quarters[i], quarters[i + 1]
        period_pair = (q_prev, q_curr)

        # Require both quarters to have valid prices for the same product
        mask = df_processed[q_prev].notna() & (df_processed[q_prev] > 0) \
             & df_processed[q_curr].notna() & (df_processed[q_curr] > 0)

        df_pair = df_processed.loc[mask].copy()
        if len(df_pair) < 10:
            continue

        # Target: log price difference
        y = np.log(df_pair[q_curr]) - np.log(df_pair[q_prev])
        X_base = df_pair[feature_cols]

        # Fill missing features conservatively with column means (within the pair)
        if X_base.isnull().any().any():
            X_base = X_base.fillna(X_base.mean())

        # Add error feature from previous period pair if available
        if i == 0:
            # First period pair: no error feature, use base features only
            X = X_base
            use_error_feature = False
        else:
            # Get previous period pair
            prev_q_prev, prev_q_curr = quarters[i-1], quarters[i]
            prev_period_pair = (prev_q_prev, prev_q_curr)
            
            if prev_period_pair in errors and len(errors[prev_period_pair]) > 0:
                # Add previous period's error as feature
                prev_errors = []
                for idx in df_pair.index:
                    if idx in errors[prev_period_pair]:
                        prev_errors.append(errors[prev_period_pair][idx])
                    else:
                        prev_errors.append(0.0)  # No previous error, use 0
                
                # Create extended features: [base_features, prev_error]
                X_extended = np.column_stack([X_base.values, prev_errors])
                
                # Handle any NaN values
                if np.isnan(X_extended).any():
                    col_means = np.nanmean(X_extended, axis=0)
                    nan_mask = np.isnan(X_extended)
                    X_extended[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                
                X = pd.DataFrame(X_extended, index=X_base.index)
                use_error_feature = True
            else:
                # No previous errors available, use base features only
                X = X_base
                use_error_feature = False

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LassoCV(cv=min(5, len(df_pair)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)

        r2 = lasso.score(X_scaled, y)
        alpha = lasso.alpha_
        n_features = int(np.sum(lasso.coef_ != 0))

        # Out-of-fold predictions to avoid mean-matching artifact
        k = min(10, max(2, len(df_pair)-1))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        y_hat_oof = np.empty_like(y.values)
        y_hat_oof[:] = np.nan
        test_r2 = []
        alphas = []
        
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            
            # Fit scaler on train only
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            
            # Inner CV to choose alpha on train
            lcv = LassoCV(cv=min(5, max(2, len(tr_idx)//2)), random_state=42, max_iter=2000)
            lcv.fit(X_tr_sc, y_tr)
            y_pred_te = lcv.predict(X_te_sc)
            
            # Store OOF predictions
            y_hat_oof[te_idx] = y_pred_te
            
            # Test R2
            ss_res = np.sum((y_te.values - y_pred_te)**2)
            ss_tot = np.sum((y_te.values - np.mean(y_te.values))**2)
            test_r2.append(1 - ss_res/ss_tot if ss_tot > 0 else np.nan)
            alphas.append(lcv.alpha_)

        jevons_actual = float(np.mean(y))
        jevons_pred = float(np.nanmean(y_hat_oof))
        r2_oof = float(np.nanmean(test_r2)) if test_r2 else np.nan
        alpha_oof = float(np.nanmean(alphas)) if alphas else np.nan

        # Store errors for next period pair
        errors[period_pair] = {}
        for idx, actual_delta, pred_delta in zip(df_pair.index, y.values, y_hat_oof):
            if not np.isnan(pred_delta):
                error = actual_delta - pred_delta
                errors[period_pair][idx] = error

        model_summary_rows.append({
            'Base Quarter': q_prev,
            'Current Quarter': q_curr,
            'Samples': len(df_pair),
            'R2_Score': r2,
            'Alpha_InSample': alpha,
            'Features_Selected': n_features,
            'R2_OOF': r2_oof,
            'Alpha_OOF_Mean': alpha_oof,
            'Jevons_Actual': jevons_actual,
            'Jevons_Predicted': jevons_pred,
            'Delta_Actual_%': jevons_actual * 100.0,
            'Delta_Predicted_%': jevons_pred * 100.0,
            'Uses_Error_Feature': use_error_feature,
        })

        # Store coefficients (only base features, error feature coefficient is last if used)
        for j, f in enumerate(feature_cols):
            coef_rows.append({
                'Base Quarter': q_prev,
                'Current Quarter': q_curr,
                'Feature': f,
                'Coefficient': lasso.coef_[j],
                'Abs_Coefficient': abs(lasso.coef_[j])
            })
        
        if use_error_feature:
            coef_rows.append({
                'Base Quarter': q_prev,
                'Current Quarter': q_curr,
                'Feature': 'prev_period_error',
                'Coefficient': lasso.coef_[-1],
                'Abs_Coefficient': abs(lasso.coef_[-1])
            })

        # Save row-level deltas for diagnostics
        id_cols = ['Company Name', 'Model Name']
        asin_cols = [col for col in df_pair.columns if 'ASIN' in col]
        id_cols = [col for col in id_cols if col in df_pair.columns] + asin_cols
        tmp = df_pair[id_cols].copy()
        tmp['Base Quarter'] = q_prev
        tmp['Current Quarter'] = q_curr
        tmp['Log_Delta_Actual'] = y.values
        tmp['Log_Delta_Predicted'] = y_hat_oof
        tmp['Delta_Predicted'] = np.exp(y_hat_oof)
        tmp['Delta_Actual'] = np.exp(y.values)
        tmp['Mean_Log_Actual'] = float(np.mean(y))
        tmp['Mean_Log_Pred_OOF'] = float(np.nanmean(y_hat_oof))
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

    print('Running delta models with error feature...')
    model_df, coef_df, delta_df = run_delta_models_with_error(df, start_from='2020 Q1')

    print('Calculating Traditional and Hedonic Jevons Indices for comparison...')
    hedonic_df, traditional_df, comparison_df = calculate_hedonic_jevons_from_deltas(delta_df)

    out = 'Lasso_Delta_Models_With_Error.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        model_df.to_excel(writer, sheet_name='Model_Summary_Delta', index=False)
        coef_df.to_excel(writer, sheet_name='Coefficients_Delta', index=False)
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name='Deltas_By_Product', index=False)
        hedonic_df.to_excel(writer, sheet_name='Hedonic_Jevons', index=False)
        traditional_df.to_excel(writer, sheet_name='Traditional_Jevons', index=False)
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

    print(f'\nWrote {out}')
    print('\n=== Delta Model with Error Feature Summary ===')
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
        
        # Count how many periods use error feature
        n_with_error = model_df['Uses_Error_Feature'].sum()
        print(f'\nPeriods using error feature: {n_with_error} / {len(model_df)}')


if __name__ == '__main__':
    main()

