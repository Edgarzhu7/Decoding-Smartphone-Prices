# Mobile Phone Price Index Analysis Project

A comprehensive analysis of mobile phone price indices using both Traditional and Hedonic regression methods. This project implements multiple modeling approaches to calculate quality-adjusted price indices, comparing results across 12 different models (6 types × 2 time dimensions).

## 📊 Project Overview

This project calculates price indices for mobile phones using:
- **Traditional Jevons Index**: Unadjusted geometric mean price index based on actual market prices
- **Hedonic Price Index**: Quality-adjusted price index using regression models to control for product characteristics

The analysis spans **2018 Q4 to 2025 Q2** and includes **152+ mobile phone models** from 17+ brands.

## 🎯 Key Objectives

1. **Calculate Price Indices**: Compute both Traditional and Hedonic Jevons indices for quarterly and annual data
2. **Quality Adjustment**: Use hedonic regression to control for product characteristics when measuring price changes
3. **Method Comparison**: Compare different modeling approaches (Lasso, Delta, Time Dummy, Error Features)
4. **Time Series Analysis**: Analyze price trends and quality-adjusted price changes over time

## 📁 Project Structure

```
ECON 495/
├── Dataset.xlsx                              # Main dataset with product features and prices
├── quarter/                                  # Quarterly models
│   ├── quarterly_jevons_index_calculator.py # Traditional Jevons Index (quarterly)
│   ├── lasso_price_prediction.py            # Basic Lasso Hedonic model
│   ├── predicted_jevons_index_calculator.py # Basic Hedonic Jevons Index
│   ├── predicted_jevons_index_with_error.py # Basic Hedonic with Error Feature
│   ├── lasso_delta_price_change.py          # Delta Model
│   ├── lasso_delta_price_change_with_error.py # Delta Model with Error Feature
│   └── lasso_time_dummy_model.py            # Time Dummy Model (OLS)
├── annual/                                   # Annual models
│   ├── annually_jevons_index_calculator.py  # Traditional Jevons Index (annual)
│   ├── lasso_price_prediction_annual.py     # Basic Lasso Hedonic model (annual)
│   ├── predicted_annual_jevons_index_calculator.py # Basic Hedonic Jevons Index (annual)
│   ├── predicted_annual_jevons_index_with_error.py # Basic Hedonic with Error Feature (annual)
│   ├── lasso_delta_price_change_annual.py   # Delta Model (annual)
│   ├── lasso_delta_price_change_annual_with_error.py # Delta Model with Error Feature (annual)
│   └── lasso_time_dummy_model_annual.py     # Time Dummy Model (annual, OLS)
├── run_all_models_and_generate_report.py    # Master script to run all models and generate PDF report
├── requirement.txt                           # Python dependencies
└── README.md                                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation

1. **Clone or download the repository**

2. **Create and activate virtual environment**:
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On macOS/Linux:
source env/bin/activate
# On Windows:
env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirement.txt
```

### Run All Models

To run all 12 models and generate a comprehensive PDF report:

```bash
python run_all_models_and_generate_report.py
```

This will:
- Run all quarterly models (6 models)
- Run all annual models (6 models)
- Generate a comprehensive PDF report with cumulative results
- Output Excel files for each model

**Output**: `Model_Results_Summary_[timestamp].pdf`

### Run Individual Models

#### Quarterly Models

```bash
# Traditional Jevons Index
cd quarter
python quarterly_jevons_index_calculator.py

# Basic Lasso Hedonic
python lasso_price_prediction.py

# Basic Hedonic with Error Feature
python predicted_jevons_index_with_error.py

# Delta Model
python lasso_delta_price_change.py

# Delta Model with Error Feature
python lasso_delta_price_change_with_error.py

# Time Dummy Model
python lasso_time_dummy_model.py
```

#### Annual Models

```bash
# Traditional Jevons Index
cd annual
python annually_jevons_index_calculator.py

# Basic Lasso Hedonic
python lasso_price_prediction_annual.py

# Basic Hedonic with Error Feature
python predicted_annual_jevons_index_with_error.py

# Delta Model
python lasso_delta_price_change_annual.py

# Delta Model with Error Feature
python lasso_delta_price_change_annual_with_error.py

# Time Dummy Model
python lasso_time_dummy_model_annual.py
```

## 📈 Models Overview

This project implements **12 models** across two dimensions:

### Model Types (6 types)

| # | Model Type | Description | Regression Method |
|---|-----------|-------------|-------------------|
| 1 | **Traditional Jevons Index** | Unadjusted geometric mean index based on actual prices | None (direct calculation) |
| 2 | **Basic Hedonic** | Hedonic regression with independent period models | LassoCV |
| 3 | **Basic Hedonic + Error** | Basic model with previous period's prediction error as feature | LassoCV |
| 4 | **Delta Model** | Direct modeling of log price changes (differences) | LassoCV |
| 5 | **Delta Model + Error** | Delta model with previous period's error as feature | LassoCV |
| 6 | **Time Dummy Model** | Pooled regression with time dummy variable | OLS (Ordinary Least Squares) |

### Time Dimensions (2 dimensions)

- **Quarterly**: Uses quarterly price data (2018 Q4 - 2025 Q2)
- **Annual**: Uses annual aggregated data (average of quarterly prices per year)

### Model Matrix

| Model Type | Quarterly | Annual |
|-----------|-----------|--------|
| Traditional Jevons Index | ✅ | ✅ |
| Basic Hedonic | ✅ | ✅ |
| Basic Hedonic + Error | ✅ | ✅ |
| Delta Model | ✅ | ✅ |
| Delta Model + Error | ✅ | ✅ |
| Time Dummy Model | ✅ | ✅ |

## 🔍 Model Details

### 1. Traditional Jevons Index

**Method**: Direct calculation without quality adjustment

**Formula**:
$$I_{t,t-1}^{Jevons} = \exp\left(\frac{1}{N} \sum_i (\ln P_{i,t} - \ln P_{i,t-1})\right)$$

**Key Features**:
- ✅ No quality adjustment
- ✅ Reflects actual market price changes
- ✅ Serves as baseline for comparison

**Prediction Range**: From one period before market entry to one period after market exit (up to 2025 Q2 / 2025)

**Files**:
- Quarterly: `quarter/quarterly_jevons_index_calculator.py`
- Annual: `annual/annually_jevons_index_calculator.py`

---

### 2. Basic Hedonic Model

**Method**: Independent Lasso regression for each period

**Formula**:
$$\ln P_{i,t} = \alpha + \sum_j \beta_j X_{ij} + \epsilon_{i,t}$$

**Key Features**:
- ✅ Quality adjustment through feature control
- ✅ Independent model for each period
- ✅ Generates PDF regression reports (coefficients, confidence intervals, p-values)
- ✅ Automatic alpha selection via LassoCV

**Prediction Range**: From one period before market entry to one period after market exit (up to 2025 Q2 / 2025)

**Output**:
- Excel file with predictions
- PDF regression summary with statistics for each period

**Files**:
- Quarterly: `quarter/lasso_price_prediction.py`
- Annual: `annual/lasso_price_prediction_annual.py`

---

### 3. Basic Hedonic + Error Feature

**Method**: Sequential training with previous period's prediction error as an additional feature

**Formula**:
$$\ln P_{i,t} = \alpha + \sum_j \beta_j X_{ij} + \gamma \cdot \text{error}_{i,t-1} + \epsilon_{i,t}$$

where $\text{error}_{i,t-1} = \ln P_{i,t-1}^{actual} - \ln P_{i,t-1}^{predicted}$

**Key Features**:
- ✅ Quality adjustment
- ✅ Captures time-series dependencies
- ✅ Error feature helps model learn prediction bias

**Prediction Range**: From one period before market entry to one period after market exit (up to 2025 Q2 / 2025)

**Files**:
- Quarterly: `quarter/predicted_jevons_index_with_error.py`
- Annual: `annual/predicted_annual_jevons_index_with_error.py`

---

### 4. Delta Model

**Method**: Direct modeling of log price changes (differences) between consecutive periods

**Formula**:
$$\ln P_{i,t+1} - \ln P_{i,t} = \alpha + \sum_j \beta_j X_{ij} + \epsilon_i$$

**Key Features**:
- ✅ Directly models price changes (not absolute prices)
- ✅ Quality adjustment through feature control
- ✅ Avoids accumulation of prediction errors
- ✅ Uses Out-of-Fold (OOF) predictions to avoid mean-matching artifacts
- ✅ More accurate Hedonic Jevons Index

**Prediction Range**: From one period before market entry to one period after market exit (up to 2025 Q2 / 2025)

**Files**:
- Quarterly: `quarter/lasso_delta_price_change.py`
- Annual: `annual/lasso_delta_price_change_annual.py`

**Why Delta Model?**
- Traditional methods predict absolute prices and then compute changes, which can accumulate errors
- Delta model directly models changes, which is more stable and avoids error accumulation
- The relationship between features and price changes may be more stable than the relationship between features and absolute prices

---

### 5. Delta Model + Error Feature

**Method**: Combines Delta model approach with error feature from previous period pair

**Formula**:
$$\ln P_{i,t+1} - \ln P_{i,t} = \alpha + \sum_j \beta_j X_{ij} + \gamma \cdot \text{error}_{i,t-1} + \epsilon_i$$

**Key Features**:
- ✅ Directly models price changes
- ✅ Quality adjustment
- ✅ Captures time-series dependencies
- ✅ Avoids error accumulation

**Prediction Range**: From one period before market entry to one period after market exit (up to 2025 Q2 / 2025)

**Files**:
- Quarterly: `quarter/lasso_delta_price_change_with_error.py`
- Annual: `annual/lasso_delta_price_change_annual_with_error.py`

---

### 6. Time Dummy Model

**Method**: Pooled regression across two consecutive periods with a time dummy variable

**Formula**:
$$\ln P_{i,t} = \alpha + \sum_j \beta_j X_{ij} + \delta \cdot D_t + \epsilon_{i,t}$$

where $D_t = 0$ for period $t$ and $D_t = 1$ for period $t+1$

**Key Features**:
- ✅ Quality adjustment
- ✅ Adjacent periods share model parameters
- ✅ Time dummy coefficient directly reflects time effect
- ✅ More efficient (one model predicts two periods)
- ✅ Uses **OLS** (Ordinary Least Squares) instead of Lasso

**Prediction Range**: From one period before market entry to one period after market exit (up to 2025 Q2 / 2025)

**Jevons Index Calculation**:
- **Traditional**: Mean of actual log price differences
- **Hedonic**: Mean of predicted log price differences (from time dummy coefficient and feature predictions)

**Files**:
- Quarterly: `quarter/lasso_time_dummy_model.py`
- Annual: `annual/lasso_time_dummy_model_annual.py`

---

## 📊 Features Used in Models

All hedonic models use **9 features** to control for product quality:

| # | Feature Name | Description | Source Column | Preprocessing |
|---|-------------|-------------|---------------|---------------|
| 1 | `is_ios` | Operating system (iOS vs Android) | `Company Name` | Apple = 1, Others = 0 |
| 2 | `mobile_weight_numeric` | Mobile weight (grams) | `Mobile Weight` | Extract numeric, default 200g |
| 3 | `ram_mem_numeric` | RAM memory (GB) | `RAM` or `Ram Mem` | Extract numeric, default 4GB |
| 4 | `front_camera_mp` | Front camera (MP) | `Front Camera` | Extract MP value, default 8MP |
| 5 | `max_mp_numeric` | Rear camera max MP | `Max_MP` | Convert to numeric, default 12MP |
| 6 | `num_cameras_numeric` | Number of cameras | `Num_Cameras` | Convert to numeric, default 2 |
| 7 | `processor_level_encoded` | Processor level | `Processor Level` | LabelEncoder: Entry Level, Midrange, Flagship |
| 8 | `battery_capacity_numeric` | Battery capacity (mAh) | `Battery Capacity` | Extract numeric, default 3000mAh |
| 9 | `screen_size_numeric` | Screen size (inches) | `Screen Size` | Extract numeric, default 6.0 inches |

**Feature Importance (from Basic Hedonic model)**:
1. **is_ios** (0.43): iOS vs Android brand premium
2. **ram_mem_numeric** (0.29): RAM capacity
3. **mobile_weight_numeric** (0.11): Mobile weight
4. **num_cameras_numeric** (0.10): Number of cameras
5. **battery_capacity_numeric** (0.10): Battery capacity
6. Other features: Front camera, max pixels, screen size, processor level

## 📋 Output Files

### Quarterly Model Outputs

| Model | Output File | Description |
|-------|------------|-------------|
| Traditional | `quarter/Quarterly_Jevons_Index_Results.xlsx` | Traditional Jevons indices |
| Basic Hedonic | `quarter/Lasso_Price_Predictions.xlsx` | Predicted prices and model summary |
| Basic Hedonic | `quarter/Lasso_Regression_Summary.pdf` | Regression statistics (coefficients, CI, p-values) |
| Basic Hedonic Jevons | `quarter/Predicted_Quarterly_Jevons_Index_Results.xlsx` | Hedonic Jevons indices |
| Basic + Error | `quarter/Predicted_Jevons_Index_With_Error_Results.xlsx` | Hedonic Jevons indices with error feature |
| Delta | `quarter/Lasso_Delta_Models.xlsx` | Traditional and Hedonic Jevons indices |
| Delta + Error | `quarter/Lasso_Delta_Models_With_Error.xlsx` | Traditional and Hedonic Jevons indices with error |
| Time Dummy | `quarter/Lasso_Time_Dummy_Models.xlsx` | Traditional and Hedonic Jevons indices |

### Annual Model Outputs

| Model | Output File | Description |
|-------|------------|-------------|
| Traditional | `annual/Annual_Jevons_Index_Results.xlsx` | Traditional Jevons indices |
| Basic Hedonic | `annual/Lasso_Price_Predictions_Annual.xlsx` | Predicted prices and model summary |
| Basic Hedonic | `annual/Lasso_Regression_Summary_Annual.pdf` | Regression statistics |
| Basic Hedonic Jevons | `annual/Predicted_Annual_Jevons_Index_Results.xlsx` | Hedonic Jevons indices |
| Basic + Error | `annual/Predicted_Annual_Jevons_Index_With_Error_Results.xlsx` | Hedonic Jevons indices with error feature |
| Delta | `annual/Lasso_Delta_Models_Annual.xlsx` | Traditional and Hedonic Jevons indices |
| Delta + Error | `annual/Lasso_Delta_Models_Annual_With_Error.xlsx` | Traditional and Hedonic Jevons indices with error |
| Time Dummy | `annual/Lasso_Time_Dummy_Models_Annual.xlsx` | Traditional and Hedonic Jevons indices |

### Master Report

Running `run_all_models_and_generate_report.py` generates:
- **`Model_Results_Summary_[timestamp].pdf`**: Comprehensive PDF report with:
  - Cumulative Jevons indices for all models
  - Comparison tables (quarterly and annual)
  - Quality adjustment effects
  - Visualizations and summary statistics

## 📊 Key Results Summary

### Quarterly Models (Cumulative from 2020 Q1)

| Model | Traditional Cumulative | Hedonic Cumulative | Quality Adjustment Effect |
|-------|----------------------|-------------------|-------------------------|
| Traditional | -1.423 (-142.31%) | - | - |
| Basic Hedonic | - | -0.676 (-67.59%) | - |
| Basic + Error | - | -0.723 (-72.26%) | - |
| Delta | -1.423 (-142.31%) | -1.411 (-141.10%) | 0.85% |
| Delta + Error | -1.423 (-142.31%) | -1.402 (-140.20%) | 1.48% |
| Time Dummy | -1.423 (-142.31%) | -0.627 (-62.68%) | 55.96% |

### Annual Models (Cumulative from 2020)

| Model | Traditional Cumulative | Hedonic Cumulative | Quality Adjustment Effect |
|-------|----------------------|-------------------|-------------------------|
| Traditional | -1.214 (-121.35%) | - | - |
| Basic Hedonic | - | -0.739 (-73.87%) | - |
| Basic + Error | - | -0.732 (-73.25%) | - |
| Delta | -1.214 (-121.35%) | -1.197 (-119.70%) | 1.36% |
| Delta + Error | -1.214 (-121.35%) | -1.197 (-119.68%) | 1.38% |
| Time Dummy | -1.214 (-121.35%) | -0.777 (-77.69%) | 35.98% |

**Note**: Negative values indicate price declines. The magnitude shows the cumulative log price change.

## 🔬 Methodology Details

### Data Processing

1. **Feature Preprocessing**:
   - Extract numeric values from text fields (RAM, weight, camera specs, etc.)
   - Handle missing values with reasonable defaults
   - Encode categorical variables (processor level, operating system)

2. **Price Transformation**:
   - All models use **log prices**: $\ln(P)$
   - This allows for percentage-based interpretation
   - Makes price changes additive: $\ln(P_{t+1}) - \ln(P_t) = \ln(P_{t+1}/P_t)$

3. **Product Lifecycle**:
   - Entry quarter/year: First period with actual price
   - Exit quarter/year: Last period with actual price
   - Prediction range: One period before entry to one period after exit (up to 2025 Q2 / 2025)

### Regression Methods

#### Lasso Regression (L1 Regularization)

- **Purpose**: Automatic feature selection and regularization
- **Method**: LassoCV with cross-validation to select optimal alpha
- **Advantages**:
  - Handles multicollinearity
  - Performs automatic feature selection
  - Prevents overfitting
- **Parameters**:
  - `random_state=42`: Ensures reproducibility
  - `max_iter=2000`: Maximum iterations
  - `cv=min(5, n_samples//2)`: Cross-validation folds

#### OLS (Ordinary Least Squares)

- **Purpose**: Used in Time Dummy Model
- **Method**: Standard OLS regression
- **Advantages**:
  - No regularization needed (pooled data provides sufficient observations)
  - Direct interpretation of coefficients
  - Standard statistical inference

### Statistical Inference

For Basic Hedonic models, bootstrap methods are used to calculate:
- **Coefficients**: Mean coefficients from bootstrap samples
- **Confidence Intervals**: 2.5th and 97.5th percentiles (95% CI)
- **P-values**: Proportion of bootstrap samples with non-zero coefficients
- **Standard Errors**: Standard deviation of bootstrap coefficients

**Post-Selection Inference**: After Lasso selects features, OLS is refit on selected features for more reliable statistical inference.

### Out-of-Fold (OOF) Predictions

Delta models use OOF predictions to avoid **mean-matching artifacts**:
- Problem: If we train and predict on the same data, predicted means can mechanically match actual means
- Solution: Use cross-validation where predictions are made on data not used in training
- Result: Unbiased estimates of model performance and price changes

## 📚 Key Concepts

### Jevons Index

The **Jevons Index** (geometric mean index) measures average price change across products:

$$I_{t,t-1}^{Jevons} = \exp\left(\frac{1}{N} \sum_i (\ln P_{i,t} - \ln P_{i,t-1})\right)$$

- **Traditional Jevons**: Uses actual prices (no quality adjustment)
- **Hedonic Jevons**: Uses predicted prices from hedonic regression (quality-adjusted)

### Hedonic Regression

**Hedonic regression** controls for product characteristics to isolate pure price changes:

- **Purpose**: Separate price changes into:
  1. Quality changes (due to changing product characteristics)
  2. Pure price changes (holding quality constant)

- **Method**: Regress prices on product features, then predict prices for all products to calculate quality-adjusted indices

### Quality Adjustment Effect

The difference between Traditional and Hedonic Jevons indices reflects the **quality adjustment effect**:

$$\text{Quality Adjustment} = I_{Traditional} - I_{Hedonic}$$

- Positive difference: Quality improvements are reducing measured price decline
- Negative difference: Quality deterioration is increasing measured price decline

## 🛠️ Technical Implementation

### Core Libraries

- **pandas**: Data manipulation and Excel I/O
- **numpy**: Numerical computations
- **scikit-learn**: Lasso regression, cross-validation, preprocessing
- **statsmodels**: OLS regression and statistical inference
- **matplotlib**: Visualization and PDF generation
- **scipy**: Statistical functions

### Code Architecture

1. **Feature Preprocessing Module** (`preprocess_features()`):
   - Standardized feature extraction across all models
   - Handles missing values and data cleaning

2. **Model Training**:
   - Each model type has its own training function
   - Consistent interface for prediction

3. **Index Calculation**:
   - Standardized Jevons index calculation
   - Handles missing values appropriately

4. **Output Generation**:
   - Excel files for detailed results
   - PDF reports for regression statistics (Basic Hedonic models)
   - Master PDF report for all models (from `run_all_models_and_generate_report.py`)

## 📝 Notes and Considerations

### Model Selection Guidelines

1. **Baseline Comparison**:
   - Use **Traditional Jevons Index** as the unadjusted baseline

2. **Standard Hedonic Analysis**:
   - **Basic Hedonic**: Standard approach, independent period models
   - **Time Dummy**: More efficient, shared parameters across adjacent periods

3. **Time Series Dependencies**:
   - **Basic + Error**: Captures time dependencies through error features
   - **Delta + Error**: Combines change modeling with time dependencies

4. **Most Accurate Quality Adjustment**:
   - **Delta Model**: Direct modeling of changes, avoids error accumulation
   - **Delta + Error**: Adds time dependency to Delta approach

### Limitations and Considerations

1. **Feature Selection**: Only 9 features are used; additional features (e.g., brand effects beyond iOS, release date, etc.) could improve models

2. **Missing Data**: Default values are used for missing features, which may introduce bias

3. **Price Data**: Limited to products with available price data; selection bias may exist

4. **Time Period**: Analysis starts from 2020 Q1 for most models (earlier periods have limited data)

5. **Model Assumptions**: 
   - Linear relationships (log-linear for prices)
   - Stable feature effects over time (may not hold for rapidly changing technology)

## 🤝 Contributing

This project is part of an Economics Honors Thesis research, focusing on price index analysis methods for electronic product markets.

For questions or issues, please refer to the documentation files in the repository:
- `all_models_comprehensive_summary.md`: Detailed model descriptions (Chinese)
- `lasso_models_comprehensive_summary.md`: Lasso model details
- `delta_model_explanation.md`: Delta model explanation

## 📄 Citation

If using methods or results from this project, please cite:

```
Mobile Phone Price Index Analysis Project (2025)
Comparative Study of Traditional Jevons Index and Hedonic Jevons Index
Quality-Adjusted Price Index Method Based on Lasso Regression and Delta Models
Economics Honors Thesis Research
```

## 📄 License

This project is for academic research purposes only.

---

**Last Updated**: January 2025

**Project Status**: Active research project

**Author**: Economics Honors Thesis Research
