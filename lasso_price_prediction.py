import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
import re

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
    df_processed['ram_mem_numeric'] = df_processed['Ram Mem'].apply(
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
            'feature_importance': dict(zip(feature_cols, lasso.coef_))
        }
        
        print(f"  Sample count: {len(quarter_data)}")
        print(f"  R² score: {r2_score:.4f}")
        print(f"  Optimal Alpha: {lasso.alpha_:.6f}")
        print(f"  Selected features: {np.sum(lasso.coef_ != 0)}/{len(feature_cols)}")
    
    return results, model_info, df_processed, predict_index

def create_prediction_excel(results, model_info, df_processed, predict_index, output_file='Lasso_Price_Predictions.xlsx'):
    """
    Create Excel file with prediction results
    """
    # Create prediction results DataFrame
    # Build predictions table only for the subset to predict
    predictions_df = df_processed.loc[predict_index, ['Company Name', 'Model Name', 'ASIN']].copy()
    
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
    
    print(f"\nPrediction results saved to: {output_file}")
    return predictions_df, model_df, importance_df

def main():
    """
    Main function
    """
    print("Reading data...")
    df = pd.read_excel('Dataset.xlsx')
    
    print(f"Dataset contains {len(df)} products")
    
    # Run quarterly Lasso regression
    results, model_info, df_processed, predict_index = run_quarterly_lasso_regression(df, start_quarter='2020 Q1')
    
    # Create prediction results Excel
    predictions_df, model_df, importance_df = create_prediction_excel(results, model_info, df_processed, predict_index)
    
    # Display model summary
    print("\n=== Model Performance Summary ===")
    print(model_df.to_string(index=False))
    
    # Display average feature importance
    print("\n=== Average Feature Importance (by absolute value) ===")
    avg_importance = importance_df.groupby('Feature')['Abs_Coefficient'].mean().sort_values(ascending=False)
    for feature, importance in avg_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    return results, model_info, predictions_df

if __name__ == "__main__":
    results, model_info, predictions_df = main()
