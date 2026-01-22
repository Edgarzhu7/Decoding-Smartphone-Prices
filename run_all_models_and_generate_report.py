#!/usr/bin/env python3
"""
Run all models and generate comprehensive PDF report
This script runs all 14 models (7 types × 2 time dimensions) and creates a summary PDF
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (11, 8.5)  # Letter size


def run_model(script_path, description):
    """Run a model script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    # Determine working directory based on script location
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir:
        # Script is in a subdirectory, run from that directory
        cwd = script_dir
    else:
        # Script is in root, run from root
        cwd = os.getcwd()
    
    try:
        result = subprocess.run(
            [sys.executable, os.path.basename(script_path)],
            cwd=cwd,  # Set working directory
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully completed: {description}")
            return True, result.stdout
        else:
            print(f"✗ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {description}")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ Error: {description} - {str(e)}")
        return False, str(e)


def extract_cumulative_results(base_dir='.'):
    """Extract cumulative Jevons indices and R² scores from output files"""
    results = {
        'quarterly': {},
        'annual': {}
    }
    r2_scores = {
        'quarterly': {},
        'annual': {}
    }
    
    # Quarterly Traditional
    # Note: Traditional includes periods from 2018 Q4, but other models start from 2020 Q1
    # For fair comparison, we should use Traditional from 2020 Q1 onwards
    try:
        trad_path = os.path.join(base_dir, 'quarter', 'Quarterly_Jevons_Index_Results.xlsx')
        # Try different sheet names
        xl_file = pd.ExcelFile(trad_path)
        sheet_name = 'Adjacent Quarters' if 'Adjacent Quarters' in xl_file.sheet_names else xl_file.sheet_names[0]
        df = pd.read_excel(trad_path, sheet_name=sheet_name)
        if 'Jevons Index' in df.columns:
            # Filter to 2020 Q1 onwards for fair comparison with other models
            if 'Base Quarter' in df.columns:
                df_filtered = df[df['Base Quarter'] >= '2020 Q1'].copy()
                cum_trad_q = df_filtered['Jevons Index'].sum()
            else:
                # If no Base Quarter column, use all (but note this in description)
                cum_trad_q = df['Jevons Index'].sum()
            results['quarterly']['Traditional'] = cum_trad_q
            # Also store full range for reference
            results['quarterly']['Traditional_All_Periods'] = df['Jevons Index'].sum()
    except Exception as e:
        pass
    
    # Quarterly Basic Hedonic - Get R² from Predicted_Quarterly_Jevons_Index_Results.xlsx (lifecycle version)
    try:
        # Get R² from Basic Hedonic model summary (lifecycle version, consistent with Jevons Index)
        basic_model_path = os.path.join(base_dir, 'quarter', 'Predicted_Quarterly_Jevons_Index_Results.xlsx')
        if os.path.exists(basic_model_path):
            xl_file = pd.ExcelFile(basic_model_path)
            if 'Model_Summary' in xl_file.sheet_names:
                df_model = pd.read_excel(basic_model_path, sheet_name='Model_Summary')
                if 'R2_Score' in df_model.columns and len(df_model) > 0:
                    avg_r2 = df_model['R2_Score'].mean()
                    r2_scores['quarterly']['Basic Hedonic'] = avg_r2
    except Exception as e:
        pass
    
    # Quarterly Basic Hedonic Jevons Index
    try:
        basic_path = os.path.join(base_dir, 'quarter', 'Predicted_Quarterly_Jevons_Index_Results.xlsx')
        xl_file = pd.ExcelFile(basic_path)
        # Try to find the right sheet (usually 'Adjacent Predicted Quarters')
        sheet_name = None
        for sheet in xl_file.sheet_names:
            if 'Adjacent' in sheet and 'Predicted' in sheet:
                sheet_name = sheet
                break
        if sheet_name is None:
            # Fallback: try any sheet with 'Adjacent' or 'Quarter'
            for sheet in xl_file.sheet_names:
                if 'Adjacent' in sheet or 'Quarter' in sheet:
                    sheet_name = sheet
                    break
        if sheet_name is None:
            sheet_name = xl_file.sheet_names[0]
        df = pd.read_excel(basic_path, sheet_name=sheet_name)
        if 'Jevons Index' in df.columns:
            cum_hed_q = df['Jevons Index'].sum()
            results['quarterly']['Basic Hedonic'] = cum_hed_q
    except Exception as e:
        pass
    
    # Quarterly Basic + Error
    try:
        error_path = os.path.join(base_dir, 'quarter', 'Predicted_Jevons_Index_With_Error_Results.xlsx')
        xl_file = pd.ExcelFile(error_path)
        # Try to find the right sheet (usually 'Adjacent Predicted Quarters')
        sheet_name = None
        for sheet in xl_file.sheet_names:
            if 'Adjacent' in sheet and 'Predicted' in sheet:
                sheet_name = sheet
                break
        if sheet_name is None:
            # Fallback: try any sheet with 'Adjacent' or 'Quarter'
            for sheet in xl_file.sheet_names:
                if 'Adjacent' in sheet or 'Quarter' in sheet:
                    sheet_name = sheet
                    break
        if sheet_name is None:
            sheet_name = xl_file.sheet_names[0]
        df = pd.read_excel(error_path, sheet_name=sheet_name)
        if 'Jevons Index' in df.columns:
            cum_err_q = df['Jevons Index'].sum()
            results['quarterly']['Basic + Error'] = cum_err_q
        
        # Extract R²
        try:
            if os.path.exists(error_path):
                xl_file_r2 = pd.ExcelFile(error_path)
                if 'Model_Summary' in xl_file_r2.sheet_names:
                    df_model = pd.read_excel(error_path, sheet_name='Model_Summary')
                    if 'R2_Score' in df_model.columns and len(df_model) > 0:
                        avg_r2 = df_model['R2_Score'].mean()
                        r2_scores['quarterly']['Basic + Error'] = avg_r2
                    elif 'R2_Score' not in df_model.columns:
                        print(f"Warning: R2_Score column not found in Model_Summary for Basic + Error")
                else:
                    print(f"Warning: Model_Summary sheet not found in {error_path}")
        except Exception as e:
            print(f"Warning: Could not extract R² for Basic + Error: {e}")
            pass
    except Exception as e:
        pass
    
    # Quarterly Delta
    try:
        delta_path = os.path.join(base_dir, 'quarter', 'Lasso_Delta_Models1.xlsx')
        df = pd.read_excel(delta_path, sheet_name='Comparison')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_delta_q = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_delta_q = df['Mean_Log_Delta_Hedonic'].sum()
            results['quarterly']['Delta Traditional'] = cum_trad_delta_q
            results['quarterly']['Delta Hedonic'] = cum_hed_delta_q
        
        # Extract R²
        try:
            df_model = pd.read_excel(delta_path, sheet_name='Model_Summary_Delta')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['quarterly']['Delta'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Quarterly Delta + Error
    try:
        delta_err_path = os.path.join(base_dir, 'quarter', 'Lasso_Delta_Models_With_Error.xlsx')
        df = pd.read_excel(delta_err_path, sheet_name='Comparison')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_delta_err_q = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_delta_err_q = df['Mean_Log_Delta_Hedonic'].sum()
            results['quarterly']['Delta+Error Traditional'] = cum_trad_delta_err_q
            results['quarterly']['Delta+Error Hedonic'] = cum_hed_delta_err_q
        
        # Extract R²
        try:
            df_model = pd.read_excel(delta_err_path, sheet_name='Model_Summary_Delta')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['quarterly']['Delta+Error'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Quarterly OLS Delta
    try:
        ols_delta_path = os.path.join(base_dir, 'quarter', 'OLS_Delta_Models.xlsx')
        df = pd.read_excel(ols_delta_path, sheet_name='Comparison')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_ols_delta_q = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_ols_delta_q = df['Mean_Log_Delta_Hedonic'].sum()
            results['quarterly']['OLS Delta Traditional'] = cum_trad_ols_delta_q
            results['quarterly']['OLS Delta Hedonic'] = cum_hed_ols_delta_q
        
        # Extract R²
        try:
            df_model = pd.read_excel(ols_delta_path, sheet_name='Model_Summary_Delta')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['quarterly']['OLS Delta'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Quarterly Time Dummy
    try:
        td_path = os.path.join(base_dir, 'quarter', 'Lasso_Time_Dummy_Models.xlsx')
        df = pd.read_excel(td_path, sheet_name='Jevons_Indices')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_td_q = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_td_q = df['Mean_Log_Delta_Hedonic'].sum()
            results['quarterly']['Time Dummy Traditional'] = cum_trad_td_q
            results['quarterly']['Time Dummy Hedonic'] = cum_hed_td_q
        
        # Extract R²
        try:
            df_model = pd.read_excel(td_path, sheet_name='Model_Summary')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['quarterly']['Time Dummy'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Annual Traditional
    try:
        trad_path = os.path.join(base_dir, 'annual', 'Annual_Jevons_Index_Results.xlsx')
        xl_file = pd.ExcelFile(trad_path)
        # Sheet name is 'Adjacent Years' (with space)
        sheet_name = 'Adjacent Years' if 'Adjacent Years' in xl_file.sheet_names else xl_file.sheet_names[0]
        df = pd.read_excel(trad_path, sheet_name=sheet_name)
        if 'Jevons Index' in df.columns:
            # Filter to 2020 onwards for fair comparison
            if 'Base Year' in df.columns:
                df_filtered = df[df['Base Year'] >= 2020].copy()
                cum_trad_a = df_filtered['Jevons Index'].sum()
            else:
                cum_trad_a = df['Jevons Index'].sum()
            results['annual']['Traditional'] = cum_trad_a
            results['annual']['Traditional_All_Periods'] = df['Jevons Index'].sum()
    except Exception as e:
        pass
    
    # Annual Basic Hedonic - Get R² from Predicted_Annual_Jevons_Index_Results.xlsx (lifecycle version)
    try:
        # Get R² from Basic Hedonic model summary (lifecycle version, consistent with Jevons Index)
        basic_model_path = os.path.join(base_dir, 'annual', 'Predicted_Annual_Jevons_Index_Results.xlsx')
        if os.path.exists(basic_model_path):
            xl_file = pd.ExcelFile(basic_model_path)
            if 'Model_Summary' in xl_file.sheet_names:
                df_model = pd.read_excel(basic_model_path, sheet_name='Model_Summary')
                if 'R2_Score' in df_model.columns and len(df_model) > 0:
                    avg_r2 = df_model['R2_Score'].mean()
                    r2_scores['annual']['Basic Hedonic'] = avg_r2
    except Exception as e:
        pass
    
    # Annual Basic Hedonic Jevons Index
    try:
        basic_path = os.path.join(base_dir, 'annual', 'Predicted_Annual_Jevons_Index_Results.xlsx')
        xl_file = pd.ExcelFile(basic_path)
        # Sheet name is 'Adjacent Predicted Years' (with space)
        sheet_name = None
        for sheet in xl_file.sheet_names:
            if 'Adjacent' in sheet and 'Predicted' in sheet:
                sheet_name = sheet
                break
        if sheet_name is None:
            sheet_name = xl_file.sheet_names[0]
        df = pd.read_excel(basic_path, sheet_name=sheet_name)
        if 'Jevons Index' in df.columns:
            cum_hed_a = df['Jevons Index'].sum()
            results['annual']['Basic Hedonic'] = cum_hed_a
    except Exception as e:
        pass
    
    # Annual Basic + Error
    try:
        error_path = os.path.join(base_dir, 'annual', 'Predicted_Annual_Jevons_Index_With_Error_Results.xlsx')
        xl_file = pd.ExcelFile(error_path)
        # Sheet name is 'Adjacent Predicted Years' (with space)
        sheet_name = None
        for sheet in xl_file.sheet_names:
            if 'Adjacent' in sheet and 'Predicted' in sheet:
                sheet_name = sheet
                break
        if sheet_name is None:
            sheet_name = xl_file.sheet_names[0]
        df = pd.read_excel(error_path, sheet_name=sheet_name)
        if 'Jevons Index' in df.columns:
            cum_err_a = df['Jevons Index'].sum()
            results['annual']['Basic + Error'] = cum_err_a
        
        # Extract R²
        try:
            if os.path.exists(error_path):
                xl_file_r2 = pd.ExcelFile(error_path)
                if 'Model_Summary' in xl_file_r2.sheet_names:
                    df_model = pd.read_excel(error_path, sheet_name='Model_Summary')
                    if 'R2_Score' in df_model.columns and len(df_model) > 0:
                        avg_r2 = df_model['R2_Score'].mean()
                        r2_scores['annual']['Basic + Error'] = avg_r2
                    elif 'R2_Score' not in df_model.columns:
                        print(f"Warning: R2_Score column not found in Model_Summary for Basic + Error (annual)")
                else:
                    print(f"Warning: Model_Summary sheet not found in {error_path}")
        except Exception as e:
            print(f"Warning: Could not extract R² for Basic + Error (annual): {e}")
            pass
    except Exception as e:
        pass
    
    # Annual Delta
    try:
        delta_path = os.path.join(base_dir, 'annual', 'Lasso_Delta_Models_Annual.xlsx')
        df = pd.read_excel(delta_path, sheet_name='Comparison')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_delta_a = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_delta_a = df['Mean_Log_Delta_Hedonic'].sum()
            results['annual']['Delta Traditional'] = cum_trad_delta_a
            results['annual']['Delta Hedonic'] = cum_hed_delta_a
        
        # Extract R²
        try:
            df_model = pd.read_excel(delta_path, sheet_name='Model_Summary_Delta')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['annual']['Delta'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Annual Delta + Error
    try:
        delta_err_path = os.path.join(base_dir, 'annual', 'Lasso_Delta_Models_Annual_With_Error.xlsx')
        df = pd.read_excel(delta_err_path, sheet_name='Comparison')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_delta_err_a = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_delta_err_a = df['Mean_Log_Delta_Hedonic'].sum()
            results['annual']['Delta+Error Traditional'] = cum_trad_delta_err_a
            results['annual']['Delta+Error Hedonic'] = cum_hed_delta_err_a
        
        # Extract R²
        try:
            df_model = pd.read_excel(delta_err_path, sheet_name='Model_Summary_Delta')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['annual']['Delta+Error'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Annual OLS Delta
    try:
        ols_delta_path = os.path.join(base_dir, 'annual', 'OLS_Delta_Models_Annual.xlsx')
        df = pd.read_excel(ols_delta_path, sheet_name='Comparison')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_ols_delta_a = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_ols_delta_a = df['Mean_Log_Delta_Hedonic'].sum()
            results['annual']['OLS Delta Traditional'] = cum_trad_ols_delta_a
            results['annual']['OLS Delta Hedonic'] = cum_hed_ols_delta_a
        
        # Extract R²
        try:
            df_model = pd.read_excel(ols_delta_path, sheet_name='Model_Summary_Delta')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['annual']['OLS Delta'] = avg_r2
        except:
            pass
    except:
        pass
    
    # Annual Time Dummy
    try:
        td_path = os.path.join(base_dir, 'annual', 'Lasso_Time_Dummy_Models_Annual.xlsx')
        df = pd.read_excel(td_path, sheet_name='Jevons_Indices')
        if 'Mean_Log_Delta_Traditional' in df.columns and 'Mean_Log_Delta_Hedonic' in df.columns:
            cum_trad_td_a = df['Mean_Log_Delta_Traditional'].sum()
            cum_hed_td_a = df['Mean_Log_Delta_Hedonic'].sum()
            results['annual']['Time Dummy Traditional'] = cum_trad_td_a
            results['annual']['Time Dummy Hedonic'] = cum_hed_td_a
        
        # Extract R²
        try:
            df_model = pd.read_excel(td_path, sheet_name='Model_Summary')
            if 'R2_Score' in df_model.columns:
                avg_r2 = df_model['R2_Score'].mean()
                r2_scores['annual']['Time Dummy'] = avg_r2
        except:
            pass
    except:
        pass
    
    return results, r2_scores


def create_summary_pdf(results_dict, r2_scores_dict, output_path='Model_Results_Summary.pdf'):
    """Create comprehensive PDF report"""
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Comprehensive Model Results Summary', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Hedonic Price Index Analysis', 
                ha='center', va='center', fontsize=18)
        fig.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.3, '14 Models: 7 Types × 2 Time Dimensions (Quarterly/Annual)', 
                ha='center', va='center', fontsize=14)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Quarterly Results Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        title = 'Quarterly Models - Cumulative Jevons Indices'
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, fontweight='bold',
                transform=ax.transAxes)
        
        y_pos = 0.85
        line_height = 0.08
        
        # Extract quarterly results
        q_results = results_dict.get('quarterly', {})
        q_r2 = r2_scores_dict.get('quarterly', {})
        
        data_rows = []
        if 'Traditional' in q_results:
            data_rows.append(('Traditional Jevons Index', q_results['Traditional'], None, None, 'No quality adjustment (2020 Q1+)'))
        if 'Basic Hedonic' in q_results:
            r2_val = q_r2.get('Basic Hedonic', None)
            data_rows.append(('Basic Hedonic', None, q_results['Basic Hedonic'], r2_val, 'Standard hedonic regression'))
        if 'Basic + Error' in q_results:
            r2_val = q_r2.get('Basic + Error', None)
            data_rows.append(('Basic + Error Feature', None, q_results['Basic + Error'], r2_val, 'With error feature'))
        if 'Delta Traditional' in q_results and 'Delta Hedonic' in q_results:
            r2_val = q_r2.get('Delta', None)
            data_rows.append(('Delta Model', q_results['Delta Traditional'], q_results['Delta Hedonic'], r2_val, 'Direct price change modeling'))
        if 'Delta+Error Traditional' in q_results and 'Delta+Error Hedonic' in q_results:
            r2_val = q_r2.get('Delta+Error', None)
            data_rows.append(('Delta + Error', q_results['Delta+Error Traditional'], q_results['Delta+Error Hedonic'], r2_val, 'Delta with error feature'))
        if 'OLS Delta Traditional' in q_results and 'OLS Delta Hedonic' in q_results:
            r2_val = q_r2.get('OLS Delta', None)
            data_rows.append(('OLS Delta Model', q_results['OLS Delta Traditional'], q_results['OLS Delta Hedonic'], r2_val, 'OLS regression (no regularization)'))
        if 'Time Dummy Traditional' in q_results and 'Time Dummy Hedonic' in q_results:
            r2_val = q_r2.get('Time Dummy', None)
            data_rows.append(('Time Dummy Model', q_results['Time Dummy Traditional'], q_results['Time Dummy Hedonic'], r2_val, 'Pooled data + time dummy'))
        
        # Create table
        table_data = [['Model', 'Traditional', 'Hedonic', 'R² (avg)', 'Description']]
        for row in data_rows:
            trad_str = f"{row[1]:.4f} ({row[1]*100:.2f}%)" if row[1] is not None else "N/A"
            hed_str = f"{row[2]:.4f} ({row[2]*100:.2f}%)" if row[2] is not None else "N/A"
            r2_str = f"{row[3]:.4f}" if row[3] is not None else "N/A"
            table_data.append([row[0], trad_str, hed_str, r2_str, row[4]])
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='left', loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.75])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Annual Results Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        title = 'Annual Models - Cumulative Jevons Indices'
        subtitle = 'All models cover 2020 to 2025 (5 periods)'
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.90, subtitle, ha='center', va='top', fontsize=10, style='italic',
                transform=ax.transAxes)
        
        # Extract annual results
        a_results = results_dict.get('annual', {})
        a_r2 = r2_scores_dict.get('annual', {})
        
        data_rows = []
        if 'Traditional' in a_results:
            data_rows.append(('Traditional Jevons Index', a_results['Traditional'], None, None, 'No quality adjustment (2020+)'))
        if 'Basic Hedonic' in a_results:
            r2_val = a_r2.get('Basic Hedonic', None)
            data_rows.append(('Basic Hedonic', None, a_results['Basic Hedonic'], r2_val, 'Standard hedonic regression'))
        if 'Basic + Error' in a_results:
            r2_val = a_r2.get('Basic + Error', None)
            data_rows.append(('Basic + Error Feature', None, a_results['Basic + Error'], r2_val, 'With error feature'))
        if 'Delta Traditional' in a_results and 'Delta Hedonic' in a_results:
            r2_val = a_r2.get('Delta', None)
            data_rows.append(('Delta Model', a_results['Delta Traditional'], a_results['Delta Hedonic'], r2_val, 'Direct price change modeling'))
        if 'Delta+Error Traditional' in a_results and 'Delta+Error Hedonic' in a_results:
            r2_val = a_r2.get('Delta+Error', None)
            data_rows.append(('Delta + Error', a_results['Delta+Error Traditional'], a_results['Delta+Error Hedonic'], r2_val, 'Delta with error feature'))
        if 'OLS Delta Traditional' in a_results and 'OLS Delta Hedonic' in a_results:
            r2_val = a_r2.get('OLS Delta', None)
            data_rows.append(('OLS Delta Model', a_results['OLS Delta Traditional'], a_results['OLS Delta Hedonic'], r2_val, 'OLS regression (no regularization)'))
        if 'Time Dummy Traditional' in a_results and 'Time Dummy Hedonic' in a_results:
            r2_val = a_r2.get('Time Dummy', None)
            data_rows.append(('Time Dummy Model', a_results['Time Dummy Traditional'], a_results['Time Dummy Hedonic'], r2_val, 'Pooled data + time dummy'))
        
        # Create table
        table_data = [['Model', 'Traditional', 'Hedonic', 'R² (avg)', 'Description']]
        for row in data_rows:
            trad_str = f"{row[1]:.4f} ({row[1]*100:.2f}%)" if row[1] is not None else "N/A"
            hed_str = f"{row[2]:.4f} ({row[2]*100:.2f}%)" if row[2] is not None else "N/A"
            r2_str = f"{row[3]:.4f}" if row[3] is not None else "N/A"
            table_data.append([row[0], trad_str, hed_str, r2_str, row[4]])
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='left', loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.75])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Comparison Chart - Quarterly
        if q_results:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            
            models = []
            traditional_vals = []
            hedonic_vals = []
            
            if 'Traditional' in q_results:
                models.append('Traditional')
                traditional_vals.append(q_results['Traditional'] * 100)
                hedonic_vals.append(np.nan)
            
            if 'Basic Hedonic' in q_results:
                models.append('Basic\nHedonic')
                traditional_vals.append(np.nan)
                hedonic_vals.append(q_results['Basic Hedonic'] * 100)
            
            if 'Basic + Error' in q_results:
                models.append('Basic\n+Error')
                traditional_vals.append(np.nan)
                hedonic_vals.append(q_results['Basic + Error'] * 100)
            
            if 'Delta Hedonic' in q_results:
                models.append('Delta\nHedonic')
                traditional_vals.append(q_results.get('Delta Traditional', np.nan) * 100)
                hedonic_vals.append(q_results['Delta Hedonic'] * 100)
            
            if 'Delta+Error Hedonic' in q_results:
                models.append('Delta\n+Error')
                traditional_vals.append(q_results.get('Delta+Error Traditional', np.nan) * 100)
                hedonic_vals.append(q_results['Delta+Error Hedonic'] * 100)
            
            if 'OLS Delta Hedonic' in q_results:
                models.append('OLS\nDelta')
                traditional_vals.append(q_results.get('OLS Delta Traditional', np.nan) * 100)
                hedonic_vals.append(q_results['OLS Delta Hedonic'] * 100)
            
            if 'Time Dummy Hedonic' in q_results:
                models.append('Time\nDummy')
                traditional_vals.append(q_results.get('Time Dummy Traditional', np.nan) * 100)
                hedonic_vals.append(q_results['Time Dummy Hedonic'] * 100)
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, traditional_vals, width, label='Traditional', color='#FF6B6B', alpha=0.8)
            bars2 = ax.bar(x + width/2, hedonic_vals, width, label='Hedonic (Quality-Adjusted)', color='#4ECDC4', alpha=0.8)
            
            ax.set_ylabel('Cumulative Price Change (%)', fontsize=12, fontweight='bold')
            ax.set_title('Quarterly Models - Cumulative Jevons Indices Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom' if height < 0 else 'top', fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Comparison Chart - Annual
        if a_results:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            
            models = []
            traditional_vals = []
            hedonic_vals = []
            
            if 'Traditional' in a_results:
                models.append('Traditional')
                traditional_vals.append(a_results['Traditional'] * 100)
                hedonic_vals.append(np.nan)
            
            if 'Basic Hedonic' in a_results:
                models.append('Basic\nHedonic')
                traditional_vals.append(np.nan)
                hedonic_vals.append(a_results['Basic Hedonic'] * 100)
            
            if 'Basic + Error' in a_results:
                models.append('Basic\n+Error')
                traditional_vals.append(np.nan)
                hedonic_vals.append(a_results['Basic + Error'] * 100)
            
            if 'Delta Hedonic' in a_results:
                models.append('Delta\nHedonic')
                traditional_vals.append(a_results.get('Delta Traditional', np.nan) * 100)
                hedonic_vals.append(a_results['Delta Hedonic'] * 100)
            
            if 'Delta+Error Hedonic' in a_results:
                models.append('Delta\n+Error')
                traditional_vals.append(a_results.get('Delta+Error Traditional', np.nan) * 100)
                hedonic_vals.append(a_results['Delta+Error Hedonic'] * 100)
            
            if 'OLS Delta Hedonic' in a_results:
                models.append('OLS\nDelta')
                traditional_vals.append(a_results.get('OLS Delta Traditional', np.nan) * 100)
                hedonic_vals.append(a_results['OLS Delta Hedonic'] * 100)
            
            if 'Time Dummy Hedonic' in a_results:
                models.append('Time\nDummy')
                traditional_vals.append(a_results.get('Time Dummy Traditional', np.nan) * 100)
                hedonic_vals.append(a_results['Time Dummy Hedonic'] * 100)
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, traditional_vals, width, label='Traditional', color='#FF6B6B', alpha=0.8)
            bars2 = ax.bar(x + width/2, hedonic_vals, width, label='Hedonic (Quality-Adjusted)', color='#4ECDC4', alpha=0.8)
            
            ax.set_ylabel('Cumulative Price Change (%)', fontsize=12, fontweight='bold')
            ax.set_title('Annual Models - Cumulative Jevons Indices Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom' if height < 0 else 'top', fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Model Descriptions
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        title = 'Model Descriptions'
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, fontweight='bold',
                transform=ax.transAxes)
        
        descriptions = [
            ('Traditional Jevons Index', 'Direct calculation using actual prices. No quality adjustment. Serves as baseline.'),
            ('Basic Hedonic', 'Standard hedonic regression. Independent Lasso model for each period. Controls for product features.'),
            ('Basic + Error Feature', 'Hedonic regression with previous period prediction error as additional feature. Captures time dependencies.'),
            ('Delta Model (Lasso)', 'Directly models log price differences between consecutive periods using Lasso. Uses OOF predictions to avoid mean-matching artifacts. Automatic feature selection.'),
            ('Delta + Error Feature', 'Combines Delta model with error feature from previous period pair. Captures both price changes and time dependencies.'),
            ('OLS Delta Model', 'Directly models log price differences using OLS (no regularization). All features included. Simpler interpretation.'),
            ('Time Dummy Model', 'Pools consecutive periods data together. Adds time dummy variable. Single model predicts both periods. More efficient parameter sharing.')
        ]
        
        y_pos = 0.85
        for model_name, desc in descriptions:
            ax.text(0.1, y_pos, f'• {model_name}:', ha='left', va='top', 
                   fontsize=11, fontweight='bold', transform=ax.transAxes)
            ax.text(0.15, y_pos - 0.03, desc, ha='left', va='top', 
                   fontsize=10, transform=ax.transAxes, wrap=True)
            y_pos -= 0.12
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ PDF report generated: {output_path}")


def main():
    """Main function to run all models and generate report"""
    
    print("="*60)
    print("Running All Models and Generating Comprehensive Report")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all models to run
    models_to_run = [
        # Quarterly models
        ('quarter/quarterly_jevons_index_calculator.py', 'Quarterly Traditional Jevons Index'),
        ('quarter/lasso_price_prediction.py', 'Quarterly Basic Lasso (Hedonic)'),
        ('quarter/predicted_jevons_index_calculator.py', 'Quarterly Basic Hedonic'),
        ('quarter/predicted_jevons_index_with_error.py', 'Quarterly Basic Hedonic + Error Feature'),
        ('quarter/lasso_delta_price_change.py', 'Quarterly Delta Model (Lasso)'),
        ('quarter/lasso_delta_price_change_with_error.py', 'Quarterly Delta Model + Error Feature'),
        ('quarter/ols_delta_price_change.py', 'Quarterly OLS Delta Model'),
        ('quarter/lasso_time_dummy_model.py', 'Quarterly Time Dummy Model'),
        
        # Annual models
        ('annual/annually_jevons_index_calculator.py', 'Annual Traditional Jevons Index'),
        ('annual/lasso_price_prediction_annual.py', 'Annual Basic Lasso (Hedonic)'),
        ('annual/predicted_annual_jevons_index_calculator.py', 'Annual Basic Hedonic'),
        ('annual/predicted_annual_jevons_index_with_error.py', 'Annual Basic Hedonic + Error Feature'),
        ('annual/lasso_delta_price_change_annual.py', 'Annual Delta Model (Lasso)'),
        ('annual/lasso_delta_price_change_annual_with_error.py', 'Annual Delta Model + Error Feature'),
        ('annual/ols_delta_price_change_annual.py', 'Annual OLS Delta Model'),
        ('annual/lasso_time_dummy_model_annual.py', 'Annual Time Dummy Model'),
    ]
    
    # Get base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
    
    # Run all models
    results = {}
    for script_path, description in models_to_run:
        # Convert to absolute path
        abs_script_path = os.path.join(base_dir, script_path) if not os.path.isabs(script_path) else script_path
        
        if os.path.exists(abs_script_path):
            success, output = run_model(abs_script_path, description)
            results[description] = {'success': success, 'output': output}
        else:
            print(f"✗ Script not found: {abs_script_path}")
            results[description] = {'success': False, 'output': 'Script not found'}
    
    # Extract cumulative results
    print("\n" + "="*60)
    print("Extracting Results...")
    print("="*60)
    
    cumulative_results, r2_scores = extract_cumulative_results(base_dir)
    
    # Generate PDF report
    print("\n" + "="*60)
    print("Generating PDF Report...")
    print("="*60)
    
    output_pdf = f"Model_Results_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    create_summary_pdf(cumulative_results, r2_scores, output_pdf)
    
    # Print summary
    print("\n" + "="*60)
    print("Execution Summary")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Models run: {successful}/{total}")
    print(f"PDF report: {output_pdf}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful < total:
        print("\nFailed models:")
        for desc, result in results.items():
            if not result['success']:
                print(f"  - {desc}")


if __name__ == '__main__':
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()

