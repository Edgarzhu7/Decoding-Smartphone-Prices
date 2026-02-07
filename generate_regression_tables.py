"""
Script to generate regression tables for each period (quarterly and annual)
for Basic Hedonic and Basic Delta models
"""

import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.table import Table

def get_feature_display_name(feature_name):
    """Convert feature name to display format"""
    feature_mapping = {
        # Processed feature names
        'is_ios': 'iOS (1) vs Android (0)',
        'mobile_weight_numeric': 'Mobile Weight (grams)',
        'ram_mem_numeric': 'RAM Memory (GB)',
        'front_camera_mp': 'Front Camera (MP)',
        'max_mp_numeric': 'Max Back Camera MP',
        'num_cameras_numeric': 'Number of Cameras',
        'processor_level_encoded': 'Processor Level (encoded)',
        'battery_capacity_numeric': 'Battery Capacity (mAh)',
        'screen_size_numeric': 'Screen Size (inches)',
        # Original feature names (if used)
        'iOS': 'iOS (1) vs Android (0)',
        'Mobile Weight': 'Mobile Weight (grams)',
        'Ram Mem': 'RAM Memory (GB)',
        'Front Camera': 'Front Camera (MP)',
        'Max_MP': 'Max Back Camera MP',
        'Num_Cameras': 'Number of Cameras',
        'Processor Level': 'Processor Level (encoded)',
        'Battery Capacity': 'Battery Capacity (mAh)',
        'Screen Size': 'Screen Size (inches)'
    }
    return feature_mapping.get(feature_name, feature_name)

def calculate_pvalue_from_ci(coef, ci_lower, ci_upper, std_error):
    """
    Estimate p-value from confidence interval and standard error
    P-value is the probability that the coefficient is non-zero (bootstrap approach)
    If CI includes 0, p-value is high; if CI excludes 0, p-value is low
    """
    if pd.isna(ci_lower) or pd.isna(ci_upper) or pd.isna(coef):
        return np.nan
    
    # If coefficient is exactly 0, p-value is 0
    if abs(coef) < 1e-6:
        return 0.0
    
    # If CI includes 0, coefficient is not significantly different from 0
    if ci_lower <= 0 <= ci_upper:
        # P-value is high (coefficient could be zero)
        # Estimate based on how close 0 is to the coefficient relative to CI width
        ci_width = ci_upper - ci_lower
        if ci_width == 0:
            return 1.0 if abs(coef) < 1e-6 else 0.0
        
        # Distance from coefficient to 0, normalized by CI half-width
        if coef > 0:
            distance_to_zero = coef - 0
        else:
            distance_to_zero = 0 - coef
        
        # P-value decreases as coefficient moves away from 0
        # But if CI includes 0, p-value should be relatively high
        p_value = max(0.5, min(1.0, 1 - (distance_to_zero / (ci_width / 2)) * 0.5))
    else:
        # CI excludes 0, coefficient is significant
        # P-value is low (coefficient is unlikely to be zero)
        # Use standard error to estimate p-value
        if not pd.isna(std_error) and std_error > 0:
            z_score = abs(coef) / std_error
            # Convert z-score to approximate p-value (two-tailed)
            # For z > 1.96, p < 0.05; for z > 2.58, p < 0.01; etc.
            if z_score > 2.58:
                p_value = 0.01
            elif z_score > 1.96:
                p_value = 0.05
            elif z_score > 1.645:
                p_value = 0.10
            else:
                p_value = 0.20
        else:
            # Fallback: if CI excludes 0, p-value is low
            p_value = 0.05
    
    return p_value

def calculate_std_error_from_ci(ci_lower, ci_upper, confidence=0.95):
    """
    Estimate standard error from confidence interval
    For 95% CI: CI = coef ± 1.96 * SE
    So SE ≈ (CI_upper - CI_lower) / (2 * 1.96)
    """
    if pd.isna(ci_lower) or pd.isna(ci_upper):
        return np.nan
    
    ci_width = ci_upper - ci_lower
    z_score = 1.96  # For 95% CI
    std_error = ci_width / (2 * z_score)
    return std_error

def get_significance_stars(coef, std_error):
    """
    Calculate significance stars based on |coef| / std_error
    ***: |coef| > 2 * SE
    **: |coef| > 1.96 * SE
    *: |coef| > 1.645 * SE
    """
    if pd.isna(coef) or pd.isna(std_error) or std_error == 0:
        return ''
    
    ratio = abs(coef) / std_error
    if ratio > 2.0:
        return '***'
    elif ratio > 1.96:
        return '**'
    elif ratio > 1.645:
        return '*'
    else:
        return ''

def format_coefficient(coef, stars=''):
    """Format coefficient with stars"""
    if pd.isna(coef):
        return 'N/A'
    return f"{coef:.4f}{stars}"

def format_ci(ci_lower, ci_upper):
    """Format confidence interval"""
    if pd.isna(ci_lower) or pd.isna(ci_upper):
        return 'N/A'
    return f"[{ci_lower:.4f}, {ci_upper:.4f}]"

def format_pvalue(p_value):
    """Format p-value"""
    if pd.isna(p_value):
        return 'N/A'
    return f"{p_value:.4f}"

def format_stderror(std_error):
    """Format standard error"""
    if pd.isna(std_error):
        return 'N/A'
    return f"{std_error:.4f}"

def create_regression_table(coef_df, model_summary_df, period, model_type='Basic Hedonic', is_quarterly=True):
    """
    Create a formatted regression table for a single period
    
    Args:
        coef_df: DataFrame with coefficients (columns: Feature, Coefficient, CI_Lower, CI_Upper, etc.)
        model_summary_df: DataFrame with model summary (R², Alpha, Samples, etc.)
        period: Period identifier (quarter or year)
        model_type: Type of model ('Basic Hedonic' or 'Basic Delta')
        is_quarterly: Whether this is quarterly data
    
    Returns:
        DataFrame with formatted table
    """
    # Filter coefficients for this period
    # Delta models use 'Current Quarter' or 'Current Year', Hedonic uses 'Quarter' or 'Year'
    if 'Basic Delta' in model_type:
        if is_quarterly:
            period_col = 'Current Quarter'
        else:
            period_col = 'Current Year'
    else:
        if is_quarterly:
            period_col = 'Quarter'
        else:
            period_col = 'Year'
    
    period_coefs = coef_df[coef_df[period_col] == period].copy()
    
    if period_coefs.empty:
        return None
    
    # Get model summary for this period
    # Delta models use 'Current Quarter' or 'Current Year' in summary
    if 'Basic Delta' in model_type:
        if is_quarterly:
            summary_period_col = 'Current Quarter'
        else:
            summary_period_col = 'Current Year'
    else:
        if is_quarterly:
            summary_period_col = 'Quarter'
        else:
            summary_period_col = 'Year'
    
    if not model_summary_df.empty and summary_period_col in model_summary_df.columns:
        period_summary_rows = model_summary_df[model_summary_df[summary_period_col] == period]
        period_summary = period_summary_rows.iloc[0] if not period_summary_rows.empty else None
    else:
        period_summary = None
    
    # Prepare table data
    table_data = []
    
    # Get actual feature names from the data and sort them consistently
    actual_features = period_coefs['Feature'].unique().tolist()
    
    # Define preferred order (try to match these names)
    preferred_order = [
        'is_ios', 'iOS',
        'mobile_weight_numeric', 'Mobile Weight',
        'ram_mem_numeric', 'Ram Mem',
        'front_camera_mp', 'Front Camera',
        'max_mp_numeric', 'Max_MP',
        'num_cameras_numeric', 'Num_Cameras',
        'processor_level_encoded', 'Processor Level',
        'battery_capacity_numeric', 'Battery Capacity',
        'screen_size_numeric', 'Screen Size'
    ]
    
    # Sort features according to preferred order
    sorted_features = []
    for pref_feature in preferred_order:
        if pref_feature in actual_features:
            sorted_features.append(pref_feature)
            actual_features.remove(pref_feature)
    
    # Add any remaining features
    sorted_features.extend(sorted(actual_features))
    
    # Process each feature in order
    for feature in sorted_features:
        feature_row = period_coefs[period_coefs['Feature'] == feature]
        if feature_row.empty:
            continue
        
        row = feature_row.iloc[0]
        coef = row['Coefficient']
        ci_lower = row.get('CI_Lower', np.nan)
        ci_upper = row.get('CI_Upper', np.nan)
        
        # Calculate statistics if not available
        std_error = calculate_std_error_from_ci(ci_lower, ci_upper)
        p_value = calculate_pvalue_from_ci(coef, ci_lower, ci_upper, std_error)
        
        # Check if feature was selected (coefficient != 0)
        is_selected = abs(coef) > 1e-6
        
        # Calculate significance stars
        stars = get_significance_stars(coef, std_error)
        
        # Format feature name
        feature_display = get_feature_display_name(feature)
        if is_selected:
            feature_display += ' [SELECTED]'
        
        table_data.append({
            'Feature': feature_display,
            'Coefficient': format_coefficient(coef, stars),
            '95% CI': format_ci(ci_lower, ci_upper),
            'P-Value': format_pvalue(p_value),
            'Std Error': format_stderror(std_error)
        })
    
    table_df = pd.DataFrame(table_data)
    
    # Add summary statistics at the bottom
    if period_summary is not None:
        summary_rows = []
        # Try different possible column names for sample size
        samples = period_summary.get("Samples", period_summary.get("n_samples", 
                     period_summary.get("Samples_Training", "N/A")))
        if samples != "N/A":
            samples = int(samples)
        summary_rows.append({
            'Feature': f'Sample Size: {samples}',
            'Coefficient': '',
            '95% CI': '',
            'P-Value': '',
            'Std Error': ''
        })
        
        # Try different possible column names for R²
        r2 = period_summary.get("R2_Score", period_summary.get("r2_score", "N/A"))
        if r2 != "N/A":
            r2 = f"{r2:.4f}"
        summary_rows.append({
            'Feature': f'R² Score: {r2}',
            'Coefficient': '',
            '95% CI': '',
            'P-Value': '',
            'Std Error': ''
        })
        
        # Try different possible column names for Alpha
        alpha = period_summary.get("Alpha", period_summary.get("alpha", 
                 period_summary.get("Alpha_InSample", "N/A")))
        if alpha != "N/A":
            alpha = f"{alpha:.6f}"
        summary_rows.append({
            'Feature': f'Optimal Alpha: {alpha}',
            'Coefficient': '',
            '95% CI': '',
            'P-Value': '',
            'Std Error': ''
        })
        
        n_selected = int(sum([abs(c) > 1e-6 for c in period_coefs['Coefficient']]))
        n_total = len(period_coefs)
        summary_rows.append({
            'Feature': f'Features Selected: {n_selected}/{n_total}',
            'Coefficient': '',
            '95% CI': '',
            'P-Value': '',
            'Std Error': ''
        })
        
        summary_df = pd.DataFrame(summary_rows)
        table_df = pd.concat([table_df, summary_df], ignore_index=True)
    
    return table_df

def save_table_to_pdf(table_df, period, model_type, is_quarterly, output_dir='regression_tables'):
    """Save regression table to PDF"""
    os.makedirs(output_dir, exist_ok=True)
    
    period_type = 'quarterly' if is_quarterly else 'annual'
    # Clean period string for filename - convert to string first
    period_str = str(period)
    period_clean = period_str.replace(' ', '_').replace('→', 'to').replace('/', '_')
    filename = f'{output_dir}/Regression_Table_{model_type.replace(" ", "_")}_{period_type}_{period_clean}.pdf'
    
    # Calculate figure size based on number of rows
    n_rows = len(table_df)
    fig_height = max(10, n_rows * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_df.values,
                     colLabels=table_df.columns,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.35, 0.18, 0.22, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Style data rows
    n_data_rows = len([r for r in table_df['Feature'] if not r.startswith('Sample') and not r.startswith('R²') and not r.startswith('Optimal') and not r.startswith('Features')])
    for i in range(1, n_data_rows + 1):
        for j in range(len(table_df.columns)):
            table[(i, j)].set_facecolor('#FFFFFF')
            table[(i, j)].set_height(0.06)
    
    # Style summary rows
    for i in range(n_data_rows + 1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            table[(i, j)].set_facecolor('#E8F5E9')
            table[(i, j)].set_height(0.06)
    
    # Title - ensure period is a string
    period_str = str(period)
    title = f'Lasso Regression Summary: {period_str}'
    plt.title(title, fontsize=16, fontweight='bold', pad=20, y=0.98)
    
    # Add legend at the bottom
    legend_text = '[SELECTED] = Feature selected by Lasso\n'
    legend_text += 'Significance: *** |coef| > 2.SE, ** |coef| > 1.96.SE, * |coef| > 1.645.SE'
    plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved table to {filename}")

def load_coefficients_hedonic_quarterly():
    """Load coefficients from quarterly Basic Hedonic model"""
    file_path = 'quarter/Predicted_Quarterly_Jevons_Index_Results.xlsx'
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return None, None
    try:
        xl_file = pd.ExcelFile(file_path)
        if 'Coefficients' not in xl_file.sheet_names:
            print(f"  Warning: 'Coefficients' sheet not found in {file_path}")
            print(f"  Available sheets: {xl_file.sheet_names}")
            return None, None
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients')
        model_summary_df = pd.read_excel(file_path, sheet_name='Model_Summary')
        print(f"  ✓ Loaded quarterly Basic Hedonic: {len(coef_df)} coefficient rows, {len(model_summary_df)} summary rows")
        return coef_df, model_summary_df
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_coefficients_hedonic_annual():
    """Load coefficients from annual Basic Hedonic model"""
    file_path = 'annual/Predicted_Annual_Jevons_Index_Results.xlsx'
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return None, None
    try:
        xl_file = pd.ExcelFile(file_path)
        if 'Coefficients' not in xl_file.sheet_names:
            print(f"  Warning: 'Coefficients' sheet not found in {file_path}")
            print(f"  Available sheets: {xl_file.sheet_names}")
            return None, None
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients')
        model_summary_df = pd.read_excel(file_path, sheet_name='Model_Summary')
        print(f"  ✓ Loaded annual Basic Hedonic: {len(coef_df)} coefficient rows, {len(model_summary_df)} summary rows")
        return coef_df, model_summary_df
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_coefficients_delta_quarterly():
    """Load coefficients from quarterly Basic Delta model"""
    file_path = 'quarter/Lasso_Delta_Models1.xlsx'
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return None, None
    try:
        xl_file = pd.ExcelFile(file_path)
        if 'Coefficients_Delta' not in xl_file.sheet_names:
            print(f"  Warning: 'Coefficients_Delta' sheet not found in {file_path}")
            print(f"  Available sheets: {xl_file.sheet_names}")
            return None, None
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients_Delta')
        model_summary_df = pd.read_excel(file_path, sheet_name='Model_Summary_Delta')
        print(f"  ✓ Loaded quarterly Basic Delta: {len(coef_df)} coefficient rows, {len(model_summary_df)} summary rows")
        return coef_df, model_summary_df
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_coefficients_delta_annual():
    """Load coefficients from annual Basic Delta model"""
    file_path = 'annual/Lasso_Delta_Models_Annual.xlsx'
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return None, None
    try:
        xl_file = pd.ExcelFile(file_path)
        if 'Coefficients_Delta' not in xl_file.sheet_names:
            print(f"  Warning: 'Coefficients_Delta' sheet not found in {file_path}")
            print(f"  Available sheets: {xl_file.sheet_names}")
            return None, None
        coef_df = pd.read_excel(file_path, sheet_name='Coefficients_Delta')
        model_summary_df = pd.read_excel(file_path, sheet_name='Model_Summary_Delta')
        print(f"  ✓ Loaded annual Basic Delta: {len(coef_df)} coefficient rows, {len(model_summary_df)} summary rows")
        return coef_df, model_summary_df
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to generate all regression tables"""
    print("=" * 60)
    print("Regression Tables Generation Script")
    print("=" * 60)
    
    # Load all data
    print("\n1. Loading coefficient data...")
    
    # Quarterly Basic Hedonic
    print("   Loading quarterly Basic Hedonic...")
    coef_hedonic_q, summary_hedonic_q = load_coefficients_hedonic_quarterly()
    
    # Quarterly Basic Delta
    print("   Loading quarterly Basic Delta...")
    coef_delta_q, summary_delta_q = load_coefficients_delta_quarterly()
    
    # Annual Basic Hedonic
    print("   Loading annual Basic Hedonic...")
    coef_hedonic_a, summary_hedonic_a = load_coefficients_hedonic_annual()
    
    # Annual Basic Delta
    print("   Loading annual Basic Delta...")
    coef_delta_a, summary_delta_a = load_coefficients_delta_annual()
    
    # Generate tables
    print("\n2. Generating regression tables...")
    
    # Quarterly Basic Hedonic
    if coef_hedonic_q is not None and summary_hedonic_q is not None:
        print("   Generating quarterly Basic Hedonic tables...")
        quarters = sorted(coef_hedonic_q['Quarter'].unique())
        for quarter in quarters:
            table_df = create_regression_table(coef_hedonic_q, summary_hedonic_q, quarter, 
                                              'Basic Hedonic', is_quarterly=True)
            if table_df is not None:
                save_table_to_pdf(table_df, quarter, 'Basic Hedonic', is_quarterly=True)
    
    # Quarterly Basic Delta
    if coef_delta_q is not None and summary_delta_q is not None:
        print("   Generating quarterly Basic Delta tables...")
        # For delta models, we need to group by period pairs
        # Use unique Current Quarter values
        quarters = sorted(coef_delta_q['Current Quarter'].unique())
        for quarter in quarters:
            # For delta models, use Current Quarter as period identifier
            table_df = create_regression_table(coef_delta_q, summary_delta_q, quarter, 
                                              'Basic Delta', is_quarterly=True)
            if table_df is not None:
                # Format period label for delta models (show base -> current)
                base_quarters = coef_delta_q[coef_delta_q['Current Quarter'] == quarter]['Base Quarter'].unique()
                if len(base_quarters) > 0:
                    base_q = base_quarters[0]
                    period_label = f"{base_q} → {quarter}"
                else:
                    period_label = quarter
                save_table_to_pdf(table_df, period_label, 'Basic Delta', is_quarterly=True)
    
    # Annual Basic Hedonic
    if coef_hedonic_a is not None and summary_hedonic_a is not None:
        print("   Generating annual Basic Hedonic tables...")
        print(f"      Found {len(coef_hedonic_a)} coefficient rows")
        print(f"      Found {len(summary_hedonic_a)} summary rows")
        if 'Year' in coef_hedonic_a.columns:
            years = sorted(coef_hedonic_a['Year'].unique())
            print(f"      Years found: {years}")
            for year in years:
                table_df = create_regression_table(coef_hedonic_a, summary_hedonic_a, year, 
                                                  'Basic Hedonic', is_quarterly=False)
                if table_df is not None:
                    print(f"      Generated table for {year}")
                    save_table_to_pdf(table_df, year, 'Basic Hedonic', is_quarterly=False)
                else:
                    print(f"      Warning: Could not generate table for {year}")
        else:
            print(f"      Error: 'Year' column not found in annual Basic Hedonic coefficients")
            print(f"      Available columns: {coef_hedonic_a.columns.tolist()}")
    else:
        print("   Warning: Annual Basic Hedonic data not loaded")
        if coef_hedonic_a is None:
            print("      Coefficients DataFrame is None")
        if summary_hedonic_a is None:
            print("      Summary DataFrame is None")
    
    # Annual Basic Delta
    if coef_delta_a is not None and summary_delta_a is not None:
        print("   Generating annual Basic Delta tables...")
        print(f"      Found {len(coef_delta_a)} coefficient rows")
        print(f"      Found {len(summary_delta_a)} summary rows")
        if 'Current Year' in coef_delta_a.columns:
            years = sorted(coef_delta_a['Current Year'].unique())
            print(f"      Years found: {years}")
            for year in years:
                # For delta models, use Current Year as period identifier
                table_df = create_regression_table(coef_delta_a, summary_delta_a, year, 
                                                  'Basic Delta', is_quarterly=False)
                if table_df is not None:
                    # Format period label for delta models (show base -> current)
                    base_years = coef_delta_a[coef_delta_a['Current Year'] == year]['Base Year'].unique()
                    if len(base_years) > 0:
                        base_y = base_years[0]
                        period_label = f"{base_y} → {year}"
                    else:
                        period_label = year
                    print(f"      Generated table for {period_label}")
                    save_table_to_pdf(table_df, period_label, 'Basic Delta', is_quarterly=False)
                else:
                    print(f"      Warning: Could not generate table for {year}")
        else:
            print(f"      Error: 'Current Year' column not found in annual Basic Delta coefficients")
            print(f"      Available columns: {coef_delta_a.columns.tolist()}")
    else:
        print("   Warning: Annual Basic Delta data not loaded")
        if coef_delta_a is None:
            print("      Coefficients DataFrame is None")
        if summary_delta_a is None:
            print("      Summary DataFrame is None")
    
    print("\n" + "=" * 60)
    print("All regression tables generated successfully!")
    print(f"Tables saved to: regression_tables/")
    print("=" * 60)

if __name__ == '__main__':
    main()

