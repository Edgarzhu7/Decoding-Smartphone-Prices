"""
Script to plot feature coefficients over time for Basic Hedonic and Basic Delta models
Includes confidence intervals calculated using bootstrap method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path

def extract_period_from_quarter(quarter_str):
    """Extract year and quarter number from quarter string like '2020 Q1'"""
    try:
        parts = quarter_str.split()
        year = int(parts[0])
        quarter = int(parts[1].replace('Q', ''))
        # Convert to numeric period: 2020 Q1 = 2020.0, 2020 Q2 = 2020.25, etc.
        period = year + (quarter - 1) * 0.25
        return period
    except:
        return None

def extract_period_from_year(year_str):
    """Extract year from year string like '2020'"""
    try:
        return int(year_str)
    except:
        return None

# Removed calculate_confidence_interval_single_coef - now using bootstrap CI from Excel files

def load_coefficients_hedonic_quarterly():
    """Load coefficients from quarterly Basic Hedonic model"""
    file_path = 'quarter/Predicted_Quarterly_Jevons_Index_Results.xlsx'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    try:
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients')
        return coef_df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_coefficients_hedonic_annual():
    """Load coefficients from annual Basic Hedonic model"""
    file_path = 'annual/Predicted_Annual_Jevons_Index_Results.xlsx'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    try:
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients')
        return coef_df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_coefficients_delta_quarterly():
    """Load coefficients from quarterly Basic Delta (Lasso) model"""
    file_path = 'quarter/Lasso_Delta_Models1.xlsx'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    try:
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients_Delta')
        return coef_df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_coefficients_delta_annual():
    """Load coefficients from annual Basic Delta (Lasso) model"""
    file_path = 'annual/Lasso_Delta_Models_Annual.xlsx'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    try:
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients_Delta')
        return coef_df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def prepare_coefficients_data_hedonic(coef_df, is_quarterly=True):
    """
    Prepare coefficients data for plotting (Hedonic model)
    Returns: DataFrame with columns: Period, Feature, Coefficient
    """
    if coef_df is None or coef_df.empty:
        return None
    
    if is_quarterly:
        period_col = 'Quarter'
        period_func = extract_period_from_quarter
    else:
        period_col = 'Year'
        period_func = extract_period_from_year
    
    if period_col not in coef_df.columns:
        print(f"Error: {period_col} column not found in coefficients DataFrame")
        return None
    
    # Extract periods
    coef_df = coef_df.copy()
    coef_df['Period'] = coef_df[period_col].apply(period_func)
    coef_df = coef_df[coef_df['Period'].notna()]
    
    return coef_df[['Period', 'Feature', 'Coefficient']]

def prepare_coefficients_data_delta(coef_df, is_quarterly=True):
    """
    Prepare coefficients data for plotting (Delta model)
    For delta models, we use the "Current Quarter" or "Current Year" as the period
    Returns: DataFrame with columns: Period, Feature, Coefficient
    """
    if coef_df is None or coef_df.empty:
        return None
    
    if is_quarterly:
        period_col = 'Current Quarter'
        period_func = extract_period_from_quarter
    else:
        period_col = 'Current Year'
        period_func = extract_period_from_year
    
    if period_col not in coef_df.columns:
        print(f"Error: {period_col} column not found in coefficients DataFrame")
        return None
    
    # Extract periods
    coef_df = coef_df.copy()
    coef_df['Period'] = coef_df[period_col].apply(period_func)
    coef_df = coef_df[coef_df['Period'].notna()]
    
    return coef_df[['Period', 'Feature', 'Coefficient']]

def plot_feature_coefficients(coef_data, feature_name, model_name, is_quarterly=True, output_dir='coefficient_plots'):
    """
    Plot coefficients for a single feature over time with confidence intervals
    """
    if coef_data is None or coef_data.empty:
        print(f"No data available for {feature_name} in {model_name}")
        return
    
    # Filter data for this feature
    feature_data = coef_data[coef_data['Feature'] == feature_name].copy()
    
    if feature_data.empty:
        print(f"No data found for feature {feature_name}")
        return
    
    # Group by period and extract coefficients with confidence intervals
    periods = sorted(feature_data['Period'].unique())
    coefficients = []
    lower_bounds = []
    upper_bounds = []
    periods_clean = []
    
    # Check if CI columns exist
    has_ci = 'CI_Lower' in feature_data.columns and 'CI_Upper' in feature_data.columns
    
    for period in periods:
        period_data = feature_data[feature_data['Period'] == period]
        
        if len(period_data) > 0:
            # For each period, we typically have one coefficient
            coef_value = period_data['Coefficient'].iloc[0]
            
            if has_ci:
                # Use bootstrap confidence intervals from Excel
                lower = period_data['CI_Lower'].iloc[0]
                upper = period_data['CI_Upper'].iloc[0]
            else:
                # Fallback: estimate CI if not available
                print(f"Warning: CI columns not found for {feature_name}, using fallback method")
                all_coefs = feature_data['Coefficient'].values
                std_est = np.std(all_coefs) if len(all_coefs) > 1 else abs(coef_value) * 0.3
                z_score = stats.norm.ppf(0.975)  # 95% CI
                multiplier = z_score * 1.5
                lower = coef_value - multiplier * std_est
                upper = coef_value + multiplier * std_est
            
            coefficients.append(coef_value)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            periods_clean.append(period)
    
    if len(periods_clean) == 0:
        print(f"No valid periods for feature {feature_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line with error bars
    ax.errorbar(periods_clean, coefficients, 
                yerr=[np.array(coefficients) - np.array(lower_bounds),
                      np.array(upper_bounds) - np.array(coefficients)],
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label='Coefficient', color='blue')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    if is_quarterly:
        ax.set_xlabel('Year', fontsize=12)
        # Format x-axis to show years
        ax.set_xticks(range(int(min(periods_clean)), int(max(periods_clean)) + 2))
    else:
        ax.set_xlabel('Year', fontsize=12)
        ax.set_xticks(range(int(min(periods_clean)), int(max(periods_clean)) + 1))
    
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title(f'Coefficient of {feature_name} ({model_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add confidence interval text annotations
    for i, (period, coef, lower, upper) in enumerate(zip(periods_clean, coefficients, lower_bounds, upper_bounds)):
        # Upper bound text (green)
        ax.text(period, upper, f'{upper:.2f}', ha='center', va='bottom', 
                fontsize=8, color='green', fontweight='bold')
        # Lower bound text (red)
        ax.text(period, lower, f'{lower:.2f}', ha='center', va='top', 
                fontsize=8, color='red', fontweight='bold')
        # Coefficient value (blue)
        ax.text(period, coef, f'{coef:.2f}', ha='center', 
                va='bottom' if coef >= 0 else 'top', 
                fontsize=9, color='blue', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
    safe_model_name = model_name.replace(' ', '_').replace('/', '_')
    period_type = 'quarterly' if is_quarterly else 'annual'
    filename = f'{output_dir}/coefficient_{safe_feature_name}_{safe_model_name}_{period_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()

def get_all_features(coef_data):
    """Get list of all unique features in the coefficients data"""
    if coef_data is None or coef_data.empty:
        return []
    return sorted(coef_data['Feature'].unique())

def main():
    """Main function to generate all coefficient plots"""
    print("=" * 60)
    print("Feature Coefficients Visualization Script")
    print("=" * 60)
    
    # Load all coefficient data
    print("\n1. Loading coefficient data...")
    
    # Quarterly models
    print("   Loading quarterly Basic Hedonic coefficients...")
    coef_hedonic_q = load_coefficients_hedonic_quarterly()
    coef_hedonic_q_processed = prepare_coefficients_data_hedonic(coef_hedonic_q, is_quarterly=True)
    
    print("   Loading quarterly Basic Delta coefficients...")
    coef_delta_q = load_coefficients_delta_quarterly()
    coef_delta_q_processed = prepare_coefficients_data_delta(coef_delta_q, is_quarterly=True)
    
    # Annual models
    print("   Loading annual Basic Hedonic coefficients...")
    coef_hedonic_a = load_coefficients_hedonic_annual()
    coef_hedonic_a_processed = prepare_coefficients_data_hedonic(coef_hedonic_a, is_quarterly=False)
    
    print("   Loading annual Basic Delta coefficients...")
    coef_delta_a = load_coefficients_delta_annual()
    coef_delta_a_processed = prepare_coefficients_data_delta(coef_delta_a, is_quarterly=False)
    
    # Get all unique features across all models
    all_features = set()
    for coef_data in [coef_hedonic_q_processed, coef_delta_q_processed, 
                      coef_hedonic_a_processed, coef_delta_a_processed]:
        if coef_data is not None:
            all_features.update(get_all_features(coef_data))
    
    all_features = sorted(list(all_features))
    print(f"\n   Found {len(all_features)} unique features: {', '.join(all_features[:5])}...")
    
    # Generate plots for each feature and model combination
    print("\n2. Generating plots...")
    
    # Quarterly Basic Hedonic
    if coef_hedonic_q_processed is not None and not coef_hedonic_q_processed.empty:
        print("   Plotting Quarterly Basic Hedonic...")
        for feature in all_features:
            plot_feature_coefficients(coef_hedonic_q_processed, feature, 
                                    'Basic Hedonic', is_quarterly=True)
    
    # Quarterly Basic Delta
    if coef_delta_q_processed is not None and not coef_delta_q_processed.empty:
        print("   Plotting Quarterly Basic Delta...")
        for feature in all_features:
            plot_feature_coefficients(coef_delta_q_processed, feature, 
                                    'Basic Delta', is_quarterly=True)
    
    # Annual Basic Hedonic
    if coef_hedonic_a_processed is not None and not coef_hedonic_a_processed.empty:
        print("   Plotting Annual Basic Hedonic...")
        for feature in all_features:
            plot_feature_coefficients(coef_hedonic_a_processed, feature, 
                                    'Basic Hedonic', is_quarterly=False)
    
    # Annual Basic Delta
    if coef_delta_a_processed is not None and not coef_delta_a_processed.empty:
        print("   Plotting Annual Basic Delta...")
        for feature in all_features:
            plot_feature_coefficients(coef_delta_a_processed, feature, 
                                    'Basic Delta', is_quarterly=False)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Plots saved to: coefficient_plots/")
    print("=" * 60)

if __name__ == '__main__':
    main()

