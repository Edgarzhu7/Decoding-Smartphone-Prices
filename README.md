# Mobile Phone Price Index Analysis Project

This project implements both Traditional Jevons Index and Hedonic Jevons Index calculations to analyze price trends in the mobile phone market.

## 📊 Project Overview

This project contains two price index calculation methods:
1. **Traditional Jevons Index**: Geometric mean price index based on actual market prices
2. **Hedonic Jevons Index**: Quality-adjusted price index based on Lasso regression predicted prices

## 🗂️ Data Structure

### Input Data
- **Dataset.xlsx**: Contains mobile phone product features and historical price data
  - Product Features: Company Name, Model Name, Mobile Weight, RAM, Front Camera, Max_MP, Num_Cameras, Processor Level, Battery Capacity, Screen Size
  - Price Data: Quarterly prices from 2018 Q4 to 2025 Q4

### Key Files
```
├── Dataset.xlsx                              # Original dataset
├── quarterly_jevons_index_calculator.py      # Traditional Jevons Index calculation (English)
├── lasso_price_prediction.py                 # Lasso regression price prediction (English)
├── predicted_jevons_index_calculator.py      # Hedonic Jevons Index calculation (English)
├── Quarterly_Jevons_Index_Results.xlsx       # Traditional index results
├── Lasso_Price_Predictions.xlsx              # Price prediction results
└── Predicted_Quarterly_Jevons_Index_Results.xlsx  # Hedonic index results
```

## 🔢 Methodology

### Traditional Jevons Index Formula
$$I_{t,t-1}^{Jevons} = \exp\left(\frac{1}{N} \sum_i (\ln P_{i,t} - \ln P_{i,t-1})\right)$$

Where:
- $P_{i,t}$ is the price of product i at time t
- $N$ is the number of products
- This index measures the geometric mean price change

### Hedonic Jevons Index Method
1. **Feature Regression**: Use Lasso regression to estimate the relationship between prices and product features
   $$\ln P_{i,t} = \alpha + \sum_j \beta_j X_{ij} + \epsilon_{i,t}$$

2. **Quality-Adjusted Prediction**: Predict prices for all products in each quarter
3. **Index Calculation**: Calculate Jevons Index using predicted prices

## 🚀 Usage Steps

### Environment Setup
```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirement.txt
```

### Step 1: Calculate Traditional Jevons Index

```bash
python quarterly_jevons_index_calculator.py
```

**Output File**: `Quarterly_Jevons_Index_Results.xlsx`

**Contains Worksheets**:
- **Adjacent Quarters**: Adjacent quarter comparisons (27 comparisons)
- **All Quarter Pairs**: All quarter pair comparisons (378 comparisons)
- **Same Quarter Across Years**: Same quarter across years comparisons (24 comparisons)
- **Summary**: Data summary

**Key Results**:
- Continuous price decline during 2019-2022
- Price trend recovery starting from 2022
- Relatively moderate quarterly price changes

### Step 2: Lasso Regression Price Prediction

```bash
python lasso_price_prediction.py
```

**Output File**: `Lasso_Price_Predictions.xlsx`

**Feature Engineering**:
- **Operating System**: iOS (1) vs Android (0)
- **Hardware Features**: Weight, RAM, Camera, Battery, Screen Size
- **Processor Level**: Categorical encoding (Entry-Level, Midrange, Flagship)

**Contains Worksheets**:
- **Predictions**: Predicted and actual prices for each product
- **Model_Summary**: Model performance statistics for each quarter
- **Feature_Importance**: Feature importance analysis
- **Feature_Description**: Feature descriptions

**Model Performance**:
- Early quarters R² > 0.90 (2020 Q1-Q2)
- Average R² ≈ 0.50-0.60
- iOS brand premium is the most important feature (coefficient 0.43)

### Step 3: Calculate Hedonic Jevons Index

```bash
python predicted_jevons_index_calculator.py
```

**Output File**: `Predicted_Quarterly_Jevons_Index_Results.xlsx`

**Contains Worksheets**:
- **Adjacent Predicted Quarters**: Adjacent predicted quarter comparisons (22 comparisons)
- **All Predicted Quarter Pairs**: All predicted quarter pair comparisons (253 comparisons)
- **Same Predicted Quarter Across Years**: Cross-year comparisons (19 comparisons)
- **Actual vs Predicted Comparison**: Actual vs predicted index comparison (22 comparisons)
- **Summary**: Data summary

## 📈 Key Findings

### Traditional Jevons Index (Based on Actual Prices)
- **Price Decline Period**: 2019-2022, annual decline of 20-27%
- **Price Recovery Period**: Post-2022, gradually narrowing decline
- **Quarterly Fluctuations**: Relatively moderate, mostly within ±10% range

### Hedonic Jevons Index (Based on Predicted Prices)
- **Quality Adjustment Effect**: Predicted prices show more volatile fluctuations
- **Maximum Changes**: 2021 Q1→Q2 (-44.54%), 2022 Q1→Q2 (+73.57%)
- **Full Sample Coverage**: Includes quality-adjusted prices for all 689 products

### Comparative Analysis
- **Trend Consistency**: Both indices show generally consistent directional trends
- **Volatility Magnitude**: Hedonic index shows greater volatility, reflecting quality adjustment effects
- **Prediction Accuracy**: Significant differences in certain quarters, particularly 2021 Q2

## 🔍 Feature Importance Ranking

1. **is_ios (0.4309)**: iOS vs Android brand premium
2. **ram_mem_numeric (0.2910)**: RAM capacity
3. **mobile_weight_numeric (0.1142)**: Mobile weight
4. **num_cameras_numeric (0.1035)**: Number of cameras
5. **battery_capacity_numeric (0.0997)**: Battery capacity
6. **Other Features**: Front camera, max pixels, screen size, processor level

## 📊 Data Statistics

- **Total Products**: 689 mobile phone models
- **Time Span**: 2018 Q4 - 2025 Q4 (29 quarters)
- **Brand Coverage**: 17 brands including Apple, Samsung, Oppo, OnePlus, Realme, Xiaomi
- **Valid Predictions**: Price predictions for 23 quarters

## 🛠️ Technical Implementation

### Core Algorithms
- **Lasso Regression**: Uses scikit-learn's LassoCV for feature selection
- **Cross Validation**: Automatic selection of optimal regularization parameter α
- **Standardization**: Feature standardization ensures model stability
- **Log Transformation**: Price log transformation ensures accurate Jevons index calculation

### Data Processing
- **Feature Extraction**: Extract numerical features from text
- **Missing Value Handling**: Fill with reasonable default values
- **Categorical Encoding**: Processor level uses LabelEncoder
- **Data Validation**: Filter invalid prices and outliers

## 📝 Citation Format

If using methods or results from this project, please cite:

```
Mobile Phone Price Index Analysis Project (2025)
Comparative Study of Traditional Jevons Index and Hedonic Jevons Index
Quality-Adjusted Price Index Method Based on Lasso Regression
```

## 🤝 Contribution

This project is part of an Economics Honors Thesis research, focusing on price index analysis methods for electronic product markets.

## 📄 License

This project is for academic research purposes only.

---

*Last Updated: October 2025*