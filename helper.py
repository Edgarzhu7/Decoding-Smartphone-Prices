# EECS 445 - Fall 2024
# Project 1 - helper.py

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import model as project1

# pd.set_option('future.no_silent_downcasting', True)


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def add_interaction_terms(df, features, year_column):
    """
    Adds interaction terms between specified features and year dummies.

    Args:
        df: DataFrame containing the data.
        features: List of feature column names to interact with the year.
        year_column: Column name for the year variable.

    Returns:
        DataFrame with interaction terms.
    """
    # Create year dummy variables
    df = pd.get_dummies(df, columns=[year_column], prefix='Year', drop_first=True)
    
    # Add interaction terms between each feature and each year dummy
    for feature in features:
        for year_dummy in [col for col in df.columns if 'Year_' in col]:
            interaction_col = f"{feature}_{year_dummy}"
            df[interaction_col] = df[feature] * df[year_dummy]
    
    return df

def get_train_test_split(csv_file: str, test_size: float = 0.2, random_state: int = 42):
    """
    Reads the given CSV file, processes the data by applying the specified encoding rules,
    fills missing values, applies MinMax normalization, and returns the train-test split.

    Args:
        csv_file: The path to the CSV file containing the dataset.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Seed used by the random number generator for reproducibility.

    Returns:
        X_train: Training features.
        y_train: Training target (Price).
        X_test: Test features.
        y_test: Test target (Price).
        feature_names: List of feature names used for training.
    """
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # df['Log Price'] = df['Adjusted USD'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    # Drop unnecessary columns based on the rules
    df = df.drop(columns=['Name', 'Model', 'Brand', 'Price', 'USD', 'Date', 'Adjusted USD'])

    # Binary encoding for "Yes"/"No" columns
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0})
    # df[binary_columns].infer_objects(copy=False)
    # df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0}).astype(int)


    # One-Hot Encoding for "Brand" and "Operating system"
    ohe = OneHotEncoder(drop=None, sparse_output=False)  # Drop first to avoid multicollinearity
    # categorical_columns = ['Brand', 'Operating system']
    categorical_columns = ['Operating system']
    
    # Apply one-hot encoding to categorical columns
    encoded_data = ohe.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_columns))

    # Drop original categorical columns after encoding
    df = df.drop(columns=categorical_columns)
    
    # Add the encoded columns back to the dataframe
    df = pd.concat([df, encoded_df], axis=1)

    # Feature engineering for Resolution (creating a new "Total Resolution" feature)
    df['Total Resolution'] = df['Resolution x'] * df['Resolution y']
    df["Pixel Density"] = np.sqrt(df["Resolution x"]**2 + df["Resolution y"]**2) / df["Screen size (inches)"]
    df = df.drop(columns=['Resolution x', 'Resolution y'])  # Optionally, drop the original resolution columns
    # df_interacted = add_interaction_terms(df, features=['Battery capacity (mAh)', 'Screen size (inches)', 'Touchscreen', 'Processor', 
    #                                         'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE', 'Number of SIMs'], year_column='year')
    # Handle missing values by imputing the median
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Separate the target variable (Price)
    X = df_imputed.drop(columns=['Log Price'])
    y = df_imputed['Log Price']

    # Normalize the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # Get the feature names
    feature_names = X.columns.tolist()

    return X_train, y_train, X_test, y_test, feature_names