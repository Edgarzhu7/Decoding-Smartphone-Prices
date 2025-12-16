import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
    non_quarter_cols = [col for col in df.columns if 'Q' not in col or not any(char.isdigit() for char in str(col))]
    # Filter to keep only relevant feature columns
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
    # The preprocess_features function expects 'Ram Mem' but dataset has 'RAM'
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


def run_delta_models_annual(df, start_from='2020'):
    """
    Run delta models for annual data
    Directly models log price differences between adjacent years
    """
    # First aggregate quarterly data to annual
    print("Aggregating quarterly data to annual averages...")
    annual_df = aggregate_quarters_to_years(df)
    
    # Prepare features
    df_processed, _ = preprocess_features(annual_df)
    feature_cols = get_feature_columns()
    
    # Identify year columns
    years = get_sorted_year_columns(df_processed)
    if start_from in years:
        years = years[years.index(start_from):]
    
    model_summary_rows = []
    coef_rows = []
    delta_rows = []
    
    for i in range(len(years) - 1):
        year_prev, year_curr = years[i], years[i + 1]
        
        # Require both years to have valid prices for the same product
        mask = df_processed[year_prev].notna() & (df_processed[year_prev] > 0) \
             & df_processed[year_curr].notna() & (df_processed[year_curr] > 0)
        
        df_pair = df_processed.loc[mask].copy()
        if len(df_pair) < 10:
            continue
        
        # Target: log price difference
        y = np.log(df_pair[year_curr]) - np.log(df_pair[year_prev])
        X = df_pair[feature_cols]
        
        # Fill missing features conservatively with column means (within the pair)
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LassoCV(cv=min(5, len(df_pair)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        r2 = lasso.score(X_scaled, y)
        alpha = lasso.alpha_
        n_features = int(np.sum(lasso.coef_ != 0))
        
        # In-sample predictions
        y_hat_in = lasso.predict(X_scaled)
        # Out-of-fold predictions to avoid mean-matching artifact
        # Use more folds for better OOF separation; fallback to LOOCV when small
        k = min(10, max(2, len(df_pair)-1))
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
        
        model_summary_rows.append({
            'Base Year': year_prev,
            'Current Year': year_curr,
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
        })
        
        for f, c in zip(feature_cols, lasso.coef_):
            coef_rows.append({
                'Base Year': year_prev,
                'Current Year': year_curr,
                'Feature': f,
                'Coefficient': c,
                'Abs_Coefficient': abs(c)
            })
        
        # Save row-level deltas for diagnostics
        # Get available identifier columns
        id_cols = ['Company Name', 'Model Name']
        asin_cols = [col for col in df_pair.columns if 'ASIN' in col]
        id_cols = [col for col in id_cols if col in df_pair.columns] + asin_cols
        tmp = df_pair[id_cols].copy()
        tmp['Base Year'] = year_prev
        tmp['Current Year'] = year_curr
        tmp['Log_Delta_Actual'] = y.values
        # Always use OOF predictions (nan-robust)
        tmp['Log_Delta_Predicted'] = y_hat_oof
        tmp['Delta_Predicted'] = np.exp(y_hat_oof)
        tmp['Delta_Actual'] = np.exp(y.values)
        # Also store means with higher precision for diagnostics
        tmp['Mean_Log_Actual'] = float(np.mean(y))
        tmp['Mean_Log_Pred_OOF'] = float(np.nanmean(y_hat_oof))
        delta_rows.append(tmp)
    
    model_df = pd.DataFrame(model_summary_rows)
    coef_df = pd.DataFrame(coef_rows)
    delta_df = pd.concat(delta_rows, axis=0) if delta_rows else pd.DataFrame()
    
    return model_df, coef_df, delta_df


def calculate_hedonic_jevons_from_deltas_annual(delta_df: pd.DataFrame):
    """
    Calculate Hedonic Jevons Index directly from predicted price changes (deltas) for annual data
    Returns both hedonic (from predicted deltas) and traditional (from actual deltas) for comparison
    """
    hedonic_results = []
    traditional_results = []
    
    # Group by year pairs
    year_pairs = delta_df.groupby(['Base Year', 'Current Year'])
    
    for (year_prev, year_curr), group in year_pairs:
        # Get predicted and actual log deltas
        log_deltas_pred = group['Log_Delta_Predicted'].dropna()
        log_deltas_actual = group['Log_Delta_Actual'].dropna()
        
        if len(log_deltas_pred) > 0:
            # Hedonic: mean log price change from predicted deltas (quality-adjusted)
            mean_log_delta_hedonic = float(np.nanmean(log_deltas_pred))
            n_products = len(log_deltas_pred)
            
            hedonic_results.append({
                'Base Year': year_prev,
                'Current Year': year_curr,
                'Period': f"{year_prev} → {year_curr}",
                'Mean_Log_Delta_Hedonic': mean_log_delta_hedonic,
                'Number_of_Products': n_products,
                'Price_Change_%_Hedonic': mean_log_delta_hedonic * 100
            })
        
        if len(log_deltas_actual) > 0:
            # Traditional: mean log price change from actual deltas (no quality adjustment)
            mean_log_delta_traditional = float(np.nanmean(log_deltas_actual))
            
            traditional_results.append({
                'Base Year': year_prev,
                'Current Year': year_curr,
                'Period': f"{year_prev} → {year_curr}",
                'Mean_Log_Delta_Traditional': mean_log_delta_traditional,
                'Price_Change_%_Traditional': mean_log_delta_traditional * 100
            })
    
    hedonic_df = pd.DataFrame(hedonic_results)
    traditional_df = pd.DataFrame(traditional_results)
    
    # Merge for comparison
    comparison_df = hedonic_df.merge(traditional_df, on=['Base Year', 'Current Year', 'Period'], how='outer')
    comparison_df['Difference'] = comparison_df['Mean_Log_Delta_Hedonic'] - comparison_df['Mean_Log_Delta_Traditional']
    comparison_df['Difference_%'] = comparison_df['Difference'] * 100
    
    return hedonic_df, traditional_df, comparison_df


def main():
    print('Reading Dataset.xlsx...')
    # Get path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'Dataset.xlsx')
    df = pd.read_excel(dataset_path)
    
    print('Running annual delta models...')
    model_df, coef_df, delta_df = run_delta_models_annual(df, start_from='2020')
    
    print('Calculating Traditional and Hedonic Jevons Indices for comparison...')
    hedonic_df, traditional_df, comparison_df = calculate_hedonic_jevons_from_deltas_annual(delta_df)
    
    out = 'Lasso_Delta_Models_Annual.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        model_df.to_excel(writer, sheet_name='Model_Summary_Delta', index=False)
        coef_df.to_excel(writer, sheet_name='Coefficients_Delta', index=False)
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name='Deltas_By_Product', index=False)
        hedonic_df.to_excel(writer, sheet_name='Hedonic_Jevons', index=False)
        traditional_df.to_excel(writer, sheet_name='Traditional_Jevons', index=False)
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
    
    print(f'\nWrote {out}')
    print('\n=== Annual Delta Model Summary ===')
    if not model_df.empty:
        print(model_df.to_string(index=False))
    
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

