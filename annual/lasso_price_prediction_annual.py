import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import statsmodels.api as sm
import warnings
import re
import sys
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Add parent directory to path to import from Quarter folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Quarter'))
from lasso_price_prediction import preprocess_features, get_feature_columns

warnings.filterwarnings('ignore')


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


def run_annual_lasso_regression(df, start_year='2020'):
    """
    Run Lasso regression for each year starting from specified year
    First aggregates quarterly data to annual averages
    """
    # Aggregate quarterly data to annual
    print("Aggregating quarterly data to annual averages...")
    annual_df = aggregate_quarters_to_years(df)
    
    # Preprocess features
    df_processed, processor_encoder = preprocess_features(annual_df)
    
    # Get all year columns
    year_columns = get_sorted_year_columns(df_processed)
    
    # Predict only for rows that have at least one observed price in any year
    predict_mask = annual_df[year_columns].notna().any(axis=1)
    predict_index = annual_df.index[predict_mask]
    
    # Find start year index
    if start_year in year_columns:
        start_idx = year_columns.index(start_year)
        target_years = year_columns[start_idx:]
    else:
        target_years = year_columns
    
    feature_cols = get_feature_columns()
    
    results = {}
    model_info = {}
    regression_stats = {}  # Store regression statistics for each year
    
    print(f"Starting annual regression analysis from {start_year}...")
    print(f"Using features: {feature_cols}")
    
    for year in target_years:
        print(f"\nProcessing year: {year}")
        
        # Get samples with price data for this year
        year_data = df_processed[df_processed[year].notna() & (df_processed[year] > 0)].copy()
        
        if len(year_data) < 10:  # Need at least 10 samples
            print(f"  Skipping {year}: insufficient samples ({len(year_data)} < 10)")
            continue
        
        # Prepare features and target variable
        X = year_data[feature_cols]
        y = np.log(year_data[year])  # log price
        
        # Check data quality
        if X.isnull().any().any() or y.isnull().any():
            print(f"  Warning: {year} has missing values, filling with mean")
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso regression (use cross-validation to select optimal alpha)
        lasso = LassoCV(cv=min(5, len(year_data)//2), random_state=42, max_iter=2000)
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
        
        regression_stats[year] = {
            'coefficients': mean_coefs,
            'lower_bound': conf_intervals[0, :],
            'upper_bound': conf_intervals[1, :],
            'p_values': p_values,
            'standard_errors': std_errors,
            'feature_names': feature_cols,
            'selected_features': selected_features  # Boolean array indicating selected features
        }
        
        # Predict only for models that have any observed price
        X_pred = df_processed.loc[predict_index, feature_cols].fillna(df_processed[feature_cols].mean())
        X_pred_scaled = scaler.transform(X_pred)
        log_predictions = lasso.predict(X_pred_scaled)
        predictions = np.exp(log_predictions)  # Convert back to price
        
        # Save results
        # Store as Series aligned to original indices for later merging
        results[year] = pd.Series(predictions, index=predict_index)
        model_info[year] = {
            'n_samples': len(year_data),
            'r2_score': r2_score,
            'alpha': lasso.alpha_,
            'n_features_selected': np.sum(lasso.coef_ != 0),
            'feature_importance': dict(zip(feature_cols, lasso.coef_))
        }
        
        print(f"  Sample count: {len(year_data)}")
        print(f"  R² score: {r2_score:.4f}")
        print(f"  Optimal Alpha: {lasso.alpha_:.6f}")
        print(f"  Selected features: {np.sum(lasso.coef_ != 0)}/{len(feature_cols)}")
    
    return results, model_info, df_processed, predict_index, regression_stats


def get_trained_models(df, start_year='2020'):
    """
    Train Lasso models for each year and return models and scalers
    This function is used by other scripts that need the trained models
    First aggregates quarterly data to annual averages
    """
    # Aggregate quarterly data to annual
    annual_df = aggregate_quarters_to_years(df)
    
    # Preprocess features
    df_processed, _ = preprocess_features(annual_df)
    
    # Get all year columns
    year_columns = get_sorted_year_columns(df_processed)
    
    if start_year in year_columns:
        start_idx = year_columns.index(start_year)
        target_years = year_columns[start_idx:]
    else:
        target_years = year_columns
    
    feature_cols = get_feature_columns()
    models = {}
    scalers = {}
    
    for year in target_years:
        year_data = df_processed[df_processed[year].notna() & (df_processed[year] > 0)].copy()
        
        if len(year_data) < 10:
            continue
        
        X = year_data[feature_cols]
        y = np.log(year_data[year])
        
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        if y.isnull().any():
            y = y.fillna(y.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LassoCV(cv=min(5, len(year_data)//2), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        models[year] = lasso
        scalers[year] = scaler
    
    return models, scalers, df_processed


def create_prediction_excel(results, model_info, df_processed, predict_index, output_file='Lasso_Price_Predictions_Annual.xlsx'):
    """
    Create Excel file with prediction results
    """
    # Create prediction results DataFrame
    # Build predictions table only for the subset to predict
    id_cols = ['Company Name', 'Model Name']
    asin_cols = [col for col in df_processed.columns if 'ASIN' in col]
    id_cols = [col for col in id_cols if col in df_processed.columns] + asin_cols
    predictions_df = df_processed.loc[predict_index, id_cols].copy()
    
    # Add predicted price columns
    for year, predictions in results.items():
        # Align by index to ensure correct row mapping
        predictions_df[f'{year}_predicted'] = predictions_df.index.map(predictions)
    
    # Add actual price columns (for comparison)
    year_columns = get_sorted_year_columns(df_processed)
    for year in results.keys():
        if year in year_columns:
            predictions_df[f'{year}_actual'] = df_processed.loc[predict_index, year]
    
    # Create model information DataFrame
    model_summary = []
    for year, info in model_info.items():
        model_summary.append({
            'Year': year,
            'Samples': info['n_samples'],
            'R2_Score': info['r2_score'],
            'Alpha': info['alpha'],
            'Features_Selected': info['n_features_selected'],
            'Total_Features': len(get_feature_columns())
        })
    
    model_df = pd.DataFrame(model_summary)
    
    # Create feature importance DataFrame
    feature_importance_data = []
    for year, info in model_info.items():
        for feature, coef in info['feature_importance'].items():
            feature_importance_data.append({
                'Year': year,
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
    
    print(f"\nPrediction results saved to: {output_file}")
    return predictions_df, model_df, importance_df

def create_regression_summary_pdf(regression_stats, model_info, output_file='Lasso_Regression_Summary_Annual.pdf'):
    """
    Create PDF report with regression statistics for each year
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
        for year in sorted(regression_stats.keys()):
            stats = regression_stats[year]
            info = model_info[year]
            
            # Create figure
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'Lasso Regression Summary: {year}', fontsize=16, fontweight='bold')
            
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
                
                # Significance stars (based on coefficient magnitude for selected features)
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
    # Get path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'Dataset.xlsx')
    df = pd.read_excel(dataset_path)
    
    print(f"Dataset contains {len(df)} products")
    
    # Run annual Lasso regression
    results, model_info, df_processed, predict_index, regression_stats = run_annual_lasso_regression(df, start_year='2020')
    
    # Create prediction results Excel
    predictions_df, model_df, importance_df = create_prediction_excel(results, model_info, df_processed, predict_index)
    
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
    
    return results, model_info, predictions_df, regression_stats


if __name__ == "__main__":
    results, model_info, predictions_df, regression_stats = main()

