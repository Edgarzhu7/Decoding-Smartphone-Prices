import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import statsmodels.api as sm
import warnings
import re
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def extract_numeric_value(value, default=0):
    """
    Extract numeric value from string
    """
    if pd.isna(value) or value == '':
        return default
    
    # Convert to string
    value_str = str(value)
    
    # Extract numbers
    numbers = re.findall(r'\d+\.?\d*', value_str)
    if numbers:
        return float(numbers[0])
    return default

def preprocess_features(df):
    """
    Preprocess feature data
    """
    df_processed = df.copy()
    
    # 1. Operating System (iOS/Android) - based on Company Name
    df_processed['is_ios'] = (df_processed['Company Name'] == 'Apple').astype(int)
    
    # 2. Mobile Weight - extract numeric value (remove 'g')
    df_processed['mobile_weight_numeric'] = df_processed['Mobile Weight'].apply(
        lambda x: extract_numeric_value(x, 200)  # default 200g
    )
    
    # 3. RAM Memory - extract numeric value (remove 'GB')
    df_processed['ram_mem_numeric'] = df_processed['RAM'].apply(
        lambda x: extract_numeric_value(x, 4)  # default 4GB
    )
    
    # 4. Front Camera - extract MP value
    df_processed['front_camera_mp'] = df_processed['Front Camera'].apply(
        lambda x: extract_numeric_value(x, 8)  # default 8MP
    )
    
    # 5. Max_MP - already numeric, but may need cleaning
    df_processed['max_mp_numeric'] = pd.to_numeric(df_processed['Max_MP'], errors='coerce').fillna(12)
    
    # 6. Num_Cameras - already numeric, but may need cleaning
    df_processed['num_cameras_numeric'] = pd.to_numeric(df_processed['Num_Cameras'], errors='coerce').fillna(2)
    
    # 7. Processor Level - categorical variable encoding
    # Normalize variants to three canonical categories: 'Entry Level', 'Midrange', 'Flagship'
    processor_encoder = LabelEncoder()
    def _canonicalize_processor_level(val: str) -> str:
        text = str(val).lower().replace('-', ' ').strip()
        # Robust keyword matching to collapse typos/variants to three classes
        if 'flag' in text:
            return 'Flagship'
        if 'mid' in text:
            return 'Midrange'
        if 'entry' in text:
            return 'Entry Level'
        # Fallback
        return 'Unknown'

    df_processed['Processor Level'] = (
        df_processed['Processor Level']
            .fillna('Unknown')
            .apply(_canonicalize_processor_level)
    )
    df_processed['processor_level_encoded'] = processor_encoder.fit_transform(df_processed['Processor Level'])
    
    # 8. Battery Capacity - extract numeric value (remove 'mAh')
    df_processed['battery_capacity_numeric'] = df_processed['Battery Capacity'].apply(
        lambda x: extract_numeric_value(str(x).replace(',', ''), 3000)  # default 3000mAh
    )
    
    # 9. Screen Size - extract numeric value (remove 'inches')
    df_processed['screen_size_numeric'] = df_processed['Screen Size'].apply(
        lambda x: extract_numeric_value(x, 6.0)  # default 6.0 inches
    )
    
    return df_processed, processor_encoder

def get_feature_columns():
    """
    Return feature column names for regression
    """
    return [
        'is_ios',                    # Operating System
        'mobile_weight_numeric',     # Mobile Weight
        'ram_mem_numeric',          # RAM Memory
        'front_camera_mp',          # Front Camera MP
        'max_mp_numeric',           # Max MP
        'num_cameras_numeric',      # Number of Cameras
        'processor_level_encoded',   # Processor Level
        'battery_capacity_numeric',  # Battery Capacity
        'screen_size_numeric'       # Screen Size
    ]

def bootstrap_lasso_statistics(X, y, alpha, n_bootstraps=1000, random_state=42):
    """
    Calculate bootstrap statistics for Lasso regression coefficients
    Uses post-selection inference: refit OLS on selected features
    
    Returns:
        mean_coefs: Mean coefficients across bootstrap samples
        conf_intervals: 95% confidence intervals (2.5%, 97.5%)
        p_values: Proportion of times coefficient is non-zero
        std_errors: Standard errors of coefficients
    """
    np.random.seed(random_state)
    n_features = X.shape[1]
    lasso = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
    boot_coefs = []
    
    for _ in range(n_bootstraps):
        # Generate bootstrap resample
        X_resampled, y_resampled = resample(X, y)
        
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
        # Fallback: return zeros if no valid bootstrap samples
        mean_coefs = np.zeros(n_features)
        conf_intervals = np.zeros((2, n_features))
        p_values = np.zeros(n_features)
        std_errors = np.zeros(n_features)
        return mean_coefs, conf_intervals, p_values, std_errors
    
    boot_coefs = np.array(boot_coefs)
    
    # Calculate mean coefficients
    mean_coefs = np.mean(boot_coefs, axis=0)
    
    # Calculate confidence intervals (2.5th and 97.5th percentiles for 95% CI)
    conf_intervals = np.percentile(boot_coefs, [2.5, 97.5], axis=0)
    
    # Calculate p-values: proportion of times coefficient is non-zero
    p_values = np.mean(boot_coefs != 0, axis=0)
    
    # Calculate standard errors
    std_errors = np.std(boot_coefs, axis=0, ddof=1)  # ddof=1 for unbiased estimate
    
    return mean_coefs, conf_intervals, p_values, std_errors

def run_quarterly_lasso_regression(df, start_quarter='2020 Q1'):
    """
    Run Lasso regression for each quarter starting from specified quarter
    """
    # Preprocess features
    df_processed, processor_encoder = preprocess_features(df)
    
    # Get all quarter columns
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    quarter_columns = sorted(quarter_columns, key=lambda x: (int(x.split()[0]), int(x.split()[1][1:])))

    # Predict only for rows that have at least one observed price in any quarter
    predict_mask = df[quarter_columns].notna().any(axis=1)
    predict_index = df.index[predict_mask]
    
    # Find start quarter index
    start_idx = quarter_columns.index(start_quarter) if start_quarter in quarter_columns else 0
    target_quarters = quarter_columns[start_idx:]
    
    feature_cols = get_feature_columns()
    
    results = {}
    model_info = {}
    regression_stats = {}  # Store regression statistics for each quarter
    
    print(f"Starting quarterly regression analysis from {start_quarter}...")
    print(f"Using features: {feature_cols}")
    
    for quarter in target_quarters:
        print(f"\nProcessing quarter: {quarter}")
        
        # Get samples with price data for this quarter
        quarter_data = df_processed[df_processed[quarter].notna() & (df_processed[quarter] > 0)].copy()
        
        if len(quarter_data) < 10:  # Need at least 10 samples
            print(f"  Skipping {quarter}: insufficient samples ({len(quarter_data)} < 10)")
            continue
        
        # Prepare features and target variable
        X = quarter_data[feature_cols]
        y = np.log(quarter_data[quarter])  # log price
        
        # Check data quality
        if X.isnull().any().any() or y.isnull().any():
            print(f"  Warning: {quarter} has missing values, filling with mean")
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso regression (use cross-validation to select optimal alpha)
        lasso = LassoCV(cv=min(5, len(quarter_data)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        # Calculate R²
        r2_score = lasso.score(X_scaled, y)
        
        # Calculate bootstrap statistics
        print(f"  Calculating bootstrap statistics (n_bootstraps=1000)...")
        mean_coefs, conf_intervals, p_values, std_errors = bootstrap_lasso_statistics(
            X_scaled, y.values, lasso.alpha_, n_bootstraps=1000, random_state=42
        )
        
        # Store regression statistics
        # Get which features were actually selected by Lasso (non-zero coefficients)
        selected_features = lasso.coef_ != 0
        
        regression_stats[quarter] = {
            'coefficients': mean_coefs,
            'lower_bound': conf_intervals[0, :],
            'upper_bound': conf_intervals[1, :],
            'p_values': p_values,
            'standard_errors': std_errors,
            'feature_names': feature_cols,
            'selected_features': selected_features  # Boolean array indicating selected features
        }
        
        # Predict only for the 152 models that have any observed price
        X_pred = df_processed.loc[predict_index, feature_cols].fillna(df_processed[feature_cols].mean())
        X_pred_scaled = scaler.transform(X_pred)
        log_predictions = lasso.predict(X_pred_scaled)
        predictions = np.exp(log_predictions)  # Convert back to price
        
        # Save results
        # Store as Series aligned to original indices for later merging
        results[quarter] = pd.Series(predictions, index=predict_index)
        model_info[quarter] = {
            'n_samples': len(quarter_data),
            'r2_score': r2_score,
            'alpha': lasso.alpha_,
            'n_features_selected': np.sum(lasso.coef_ != 0),
            'feature_importance': dict(zip(feature_cols, lasso.coef_)),
            'log_predictions': pd.Series(log_predictions, index=predict_index)  # Save predicted log prices
        }
        
        print(f"  Sample count: {len(quarter_data)}")
        print(f"  R² score: {r2_score:.4f}")
        print(f"  Optimal Alpha: {lasso.alpha_:.6f}")
        print(f"  Selected features: {np.sum(lasso.coef_ != 0)}/{len(feature_cols)}")
    
    return results, model_info, df_processed, predict_index, regression_stats

def calculate_price_change_r2(model_info, df_processed, quarter_columns):
    """
    Calculate price-change R² for each adjacent interval using a regression:
        Δ log p_actual  =  β0  +  β1 * Δ log p_predicted  +  ε
    and report the R² of this regression.

    This follows the suggestion:
      - Instead of R² = 1 - SSE/SST on (actual Δ - predicted Δ),
        run a regression of actual log price changes on predicted log price changes
        from the hedonic level regressions, and use that regression's R².

    Args:
        model_info: dict keyed by quarter, containing at least 'log_predictions'
        df_processed: dataframe with actual prices by quarter
        quarter_columns: list of quarter column names in chronological order

    Returns:
        Dictionary mapping interval label (e.g., '2020 Q1 → 2020 Q2')
        to a dict with R², sample size, and regression slope.
    """
    interval_r2 = {}

    # Ensure quarters are in chronological order
    sorted_quarters = sorted(
        quarter_columns,
        key=lambda x: (int(x.split()[0]), int(x.split()[1][1:]))
    )

    # Iterate through adjacent quarter pairs
    for i in range(len(sorted_quarters) - 1):
        q_prev = sorted_quarters[i]
        q_curr = sorted_quarters[i + 1]

        # Need both quarters to have trained models and stored predictions
        if q_prev not in model_info or q_curr not in model_info:
            continue
        if 'log_predictions' not in model_info[q_prev] or 'log_predictions' not in model_info[q_curr]:
            continue

        # Predicted log prices
        log_pred_prev = model_info[q_prev]['log_predictions']
        log_pred_curr = model_info[q_curr]['log_predictions']

        # Actual prices
        actual_prev = df_processed[q_prev]
        actual_curr = df_processed[q_curr]

        # Indices where we have predicted logs for BOTH quarters
        pred_indices_prev = set(log_pred_prev.index)
        pred_indices_curr = set(log_pred_curr.index)
        pred_common = pred_indices_prev & pred_indices_curr

        # Indices where we have valid actual prices for BOTH quarters
        actual_mask = (
            actual_prev.notna() & (actual_prev > 0) &
            actual_curr.notna() & (actual_curr > 0)
        )
        actual_indices = set(df_processed.index[actual_mask])

        # Final common set: must have both predicted and actual prices in both quarters
        common_indices = sorted(list(pred_common & actual_indices))

        if len(common_indices) < 5:
            # Too few observations to run a meaningful regression
            print(f"  {q_prev} → {q_curr}: skipped (common samples n={len(common_indices)} < 5)")
            continue

        # Extract values
        log_pred_prev_vals = log_pred_prev.loc[common_indices].values
        log_pred_curr_vals = log_pred_curr.loc[common_indices].values
        actual_prev_vals = actual_prev.loc[common_indices].values
        actual_curr_vals = actual_curr.loc[common_indices].values

        # Construct actual and predicted log price changes
        delta_actual = np.log(actual_curr_vals) - np.log(actual_prev_vals)
        delta_pred = log_pred_curr_vals - log_pred_prev_vals

        # For reporting
        mean_actual_delta = float(np.mean(delta_actual))
        mean_predicted_delta = float(np.mean(delta_pred))

        # Run OLS: Δ log p_actual on Δ log p_pred
        X = sm.add_constant(delta_pred)  # add intercept
        y = delta_actual
        ols_model = sm.OLS(y, X).fit()
        r2 = float(ols_model.rsquared)

        interval_label = f"{q_prev} → {q_curr}"
        interval_r2[interval_label] = {
            'r2': r2,
            'n_samples': len(common_indices),
            'mean_actual_delta': mean_actual_delta,
            'mean_predicted_delta': mean_predicted_delta,
            'beta0': float(ols_model.params[0]),
            'beta1': float(ols_model.params[1]),
        }

        print(
            f"  {interval_label}: R² (price-change, regression) = {r2:.4f} "
            f"(n={len(common_indices)}, beta1={ols_model.params[1]:.4f})"
        )

    return interval_r2

def get_trained_models(df, start_quarter='2020 Q1'):
    """
    Train Lasso models for each quarter and return models and scalers
    This function is used by other scripts that need the trained models
    """
    df_processed, _ = preprocess_features(df)
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    quarter_columns = sorted(quarter_columns, key=lambda x: (int(x.split()[0]), int(x.split()[1][1:])))
    
    start_idx = quarter_columns.index(start_quarter) if start_quarter in quarter_columns else 0
    target_quarters = quarter_columns[start_idx:]
    
    feature_cols = get_feature_columns()
    models = {}
    scalers = {}
    
    for quarter in target_quarters:
        quarter_data = df_processed[df_processed[quarter].notna() & (df_processed[quarter] > 0)].copy()
        
        if len(quarter_data) < 10:
            continue
        
        X = quarter_data[feature_cols]
        y = np.log(quarter_data[quarter])
        
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        if y.isnull().any():
            y = y.fillna(y.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LassoCV(cv=min(5, len(quarter_data)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        models[quarter] = lasso
        scalers[quarter] = scaler
    
    return models, scalers, df_processed

def create_prediction_excel(results, model_info, df_processed, predict_index, interval_r2=None, output_file='Lasso_Price_Predictions.xlsx'):
    """
    Create Excel file with prediction results
    """
    # Create prediction results DataFrame
    # Build predictions table only for the subset to predict
    predictions_df = df_processed.loc[predict_index, ['Company Name', 'Model Name', 'ASIN1']].copy()
    
    # Add predicted price columns
    for quarter, predictions in results.items():
        # Align by index to ensure correct row mapping
        predictions_df[f'{quarter}_predicted'] = predictions_df.index.map(predictions)
    
    # Add actual price columns (for comparison)
    quarter_columns = [col for col in df_processed.columns if 'Q' in col and any(char.isdigit() for char in col)]
    for quarter in results.keys():
        if quarter in quarter_columns:
            predictions_df[f'{quarter}_actual'] = df_processed.loc[predict_index, quarter]
    
    # Create model information DataFrame
    model_summary = []
    for quarter, info in model_info.items():
        model_summary.append({
            'Quarter': quarter,
            'Samples': info['n_samples'],
            'R2_Score': info['r2_score'],
            'Alpha': info['alpha'],
            'Features_Selected': info['n_features_selected'],
            'Total_Features': len(get_feature_columns())
        })
    
    model_df = pd.DataFrame(model_summary)
    
    # Add summary row with overall R² (average across all quarters)
    if len(model_df) > 0:
        overall_r2 = model_df['R2_Score'].mean()
        summary_row = pd.DataFrame({
            'Quarter': ['Overall (Average)'],
            'Samples': [model_df['Samples'].sum()],
            'R2_Score': [overall_r2],
            'Alpha': [model_df['Alpha'].mean()],
            'Features_Selected': [model_df['Features_Selected'].mean()],
            'Total_Features': [len(get_feature_columns())]
        })
        model_df = pd.concat([model_df, summary_row], ignore_index=True)
    
    # Create feature importance DataFrame
    feature_importance_data = []
    for quarter, info in model_info.items():
        for feature, coef in info['feature_importance'].items():
            feature_importance_data.append({
                'Quarter': quarter,
                'Feature': feature,
                'Coefficient': coef,
                'Abs_Coefficient': abs(coef)
            })
    
    importance_df = pd.DataFrame(feature_importance_data)
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Prediction results
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Model summary
        model_df.to_excel(writer, sheet_name='Model_Summary', index=False)
        
        # Feature importance
        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # Feature description
        feature_description = pd.DataFrame({
            'Feature': get_feature_columns(),
            'Description': [
                'iOS (1) vs Android (0)',
                'Mobile Weight (grams)',
                'RAM Memory (GB)',
                'Front Camera (MP)',
                'Max Back Camera MP',
                'Number of Cameras',
                'Processor Level (encoded)',
                'Battery Capacity (mAh)',
                'Screen Size (inches)'
            ]
        })
        feature_description.to_excel(writer, sheet_name='Feature_Description', index=False)
        
        # Add price-change R² summary if available
        if interval_r2 is not None and len(interval_r2) > 0:
            price_change_r2_data = []
            for interval, stats in interval_r2.items():
                price_change_r2_data.append({
                    'Interval': interval,
                    'R2_Price_Change': stats['r2'],
                    'Samples': stats['n_samples'],
                    'Mean_Actual_Delta': stats['mean_actual_delta'],
                    'Mean_Predicted_Delta': stats['mean_predicted_delta']
                })
            
            price_change_df = pd.DataFrame(price_change_r2_data)
            
            # Overall price-change R² = simple average across all intervals
            overall_price_change_r2 = price_change_df['R2_Price_Change'].mean()
            summary_row = pd.DataFrame({
                'Interval': ['Overall (Average)'],
                'R2_Price_Change': [overall_price_change_r2],
                'Samples': [price_change_df['Samples'].sum()],
                'Mean_Actual_Delta': [price_change_df['Mean_Actual_Delta'].mean()],
                'Mean_Predicted_Delta': [price_change_df['Mean_Predicted_Delta'].mean()]
            })
            price_change_df = pd.concat([price_change_df, summary_row], ignore_index=True)
            
            price_change_df.to_excel(writer, sheet_name='Price_Change_R2', index=False)
    
    print(f"\nPrediction results saved to: {output_file}")
    return predictions_df, model_df, importance_df

def create_regression_summary_pdf(regression_stats, model_info, output_file='Lasso_Regression_Summary.pdf'):
    """
    Create PDF report with regression statistics for each quarter
    """
    feature_cols = get_feature_columns()
    feature_descriptions = {
        'is_ios': 'iOS (1) vs Android (0)',
        'mobile_weight_numeric': 'Mobile Weight (grams)',
        'ram_mem_numeric': 'RAM Memory (GB)',
        'front_camera_mp': 'Front Camera (MP)',
        'max_mp_numeric': 'Max Back Camera MP',
        'num_cameras_numeric': 'Number of Cameras',
        'processor_level_encoded': 'Processor Level (encoded)',
        'battery_capacity_numeric': 'Battery Capacity (mAh)',
        'screen_size_numeric': 'Screen Size (inches)'
    }
    
    with PdfPages(output_file) as pdf:
        for quarter in sorted(regression_stats.keys()):
            stats = regression_stats[quarter]
            info = model_info[quarter]
            
            # Create figure
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'Lasso Regression Summary: {quarter}', fontsize=16, fontweight='bold')
            
            # Model information
            info_text = f"""
Model Information:
• Sample Size: {info['n_samples']}
• R² Score: {info['r2_score']:.4f}
• Optimal Alpha: {info['alpha']:.6f}
• Features Selected: {info['n_features_selected']}/{len(feature_cols)}
            """
            
            # Create regression table
            selected_features = stats['selected_features']
            table_data = []
            for i, feature in enumerate(feature_cols):
                coef = stats['coefficients'][i]
                lower = stats['lower_bound'][i]
                upper = stats['upper_bound'][i]
                pval = stats['p_values'][i]
                se = stats['standard_errors'][i]
                is_selected = selected_features[i]
                
                # Mark selected features with [SELECTED]
                feature_name = feature_descriptions.get(feature, feature)
                if is_selected:
                    feature_name = f"{feature_name} [SELECTED]"
                
                # Significance stars (based on p-value, but note: p-value here is selection frequency)
                # For selected features, we show significance based on coefficient magnitude
                if is_selected:
                    # For selected features, check if coefficient is significantly different from 0
                    if lower > 0 or upper < 0:  # CI doesn't contain 0
                        if abs(coef) > 2 * se:  # Rough significance check
                            sig = '***'
                        elif abs(coef) > 1.96 * se:
                            sig = '**'
                        elif abs(coef) > 1.645 * se:
                            sig = '*'
                        else:
                            sig = ''
                    else:
                        sig = ''
                else:
                    sig = ''  # Not selected, no significance
                
                table_data.append([
                    feature_name,
                    f'{coef:.4f}{sig}',
                    f'[{lower:.4f}, {upper:.4f}]',
                    f'{pval:.4f}',
                    f'{se:.4f}'
                ])
            
            # Create table
            ax = fig.add_subplot(111)
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(
                cellText=table_data,
                colLabels=['Feature', 'Coefficient', '95% CI', 'P-Value', 'Std Error'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.15, 0.25, 0.15, 0.15]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style selected features (highlight in light green)
            for i in range(1, len(table_data) + 1):
                feature_name = table_data[i-1][0]
                if '[SELECTED]' in feature_name:
                    for j in range(5):
                        table[(i, j)].set_facecolor('#E8F5E9')  # Light green for selected features
            
            # Add info text
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add legend
            legend_text = "[SELECTED] = Feature selected by Lasso | Significance: *** |coef|>2*SE, ** |coef|>1.96*SE, * |coef|>1.645*SE"
            fig.text(0.5, 0.95, legend_text, ha='center', fontsize=8, style='italic')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"\nRegression summary PDF saved to: {output_file}")
    return output_file

def main():
    """
    Main function
    """
    print("Reading data...")
    df = pd.read_excel('../Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Run quarterly Lasso regression
    results, model_info, df_processed, predict_index, regression_stats = run_quarterly_lasso_regression(df, start_quarter='2020 Q1')
    
    # Calculate price-change R² for each interval
    print("\n=== Calculating Price-Change R² for Each Interval ===")
    quarter_columns = [col for col in df.columns if 'Q' in col and any(char.isdigit() for char in col)]
    interval_r2 = calculate_price_change_r2(model_info, df_processed, quarter_columns)
    
    # Create prediction results Excel
    predictions_df, model_df, importance_df = create_prediction_excel(results, model_info, df_processed, predict_index, interval_r2=interval_r2)
    
    # Create regression summary PDF
    create_regression_summary_pdf(regression_stats, model_info)
    
    # Display model summary
    print("\n=== Model Performance Summary ===")
    print(model_df.to_string(index=False))
    
    # Display average feature importance
    print("\n=== Average Feature Importance (by absolute value) ===")
    avg_importance = importance_df.groupby('Feature')['Abs_Coefficient'].mean().sort_values(ascending=False)
    for feature, importance in avg_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    # Display price-change R² summary
    print("\n=== Price-Change R² Summary ===")
    for interval, stats in interval_r2.items():
        print(f"{interval}: R² = {stats['r2']:.4f}, n = {stats['n_samples']}, "
              f"Mean Actual Δ = {stats['mean_actual_delta']:.4f}, "
              f"Mean Predicted Δ = {stats['mean_predicted_delta']:.4f}")
    
    # Calculate and display overall R² values
    print(f"\n=== Overall R² Summary ===")
    if len(model_info) > 0:
        # Overall Level R² (average across all quarters)
        level_r2_values = [info['r2_score'] for info in model_info.values()]
        overall_level_r2 = np.mean(level_r2_values)
        print(f"Overall Level R² (average across all quarters): {overall_level_r2:.4f}")
    
    if len(interval_r2) > 0:
        # Overall Price-change R² (simple average across all intervals)
        price_change_r2_values = [stats['r2'] for stats in interval_r2.values() if not np.isnan(stats['r2'])]
        if len(price_change_r2_values) > 0:
            overall_price_change_r2 = np.mean(price_change_r2_values)
            print(f"Overall Price-Change R² (average across all intervals): {overall_price_change_r2:.4f}")
    
    return results, model_info, predictions_df, regression_stats, interval_r2

if __name__ == "__main__":
    results, model_info, predictions_df, regression_stats, interval_r2 = main()
