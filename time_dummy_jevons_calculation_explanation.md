# Time Dummy Model: Jevons Index Calculation Explanation

## Overview

The Time Dummy model calculates two cumulative Jevons indices:
- **Traditional Jevons Index**: -1.423115 (-142.31%)
- **Hedonic Jevons Index**: -0.626765 (-62.68%)

This document explains how these values are calculated.

---

## Step-by-Step Calculation Process

### Step 1: Model Training (for each quarter pair)

For each consecutive quarter pair (Q1 → Q2), the model:

1. **Pools data** from both quarters
2. **Adds time dummy variable** (0 for Q1, 1 for Q2)
3. **Trains a single Lasso model**: `log(price) = f(features, time_dummy)`

**Example**: For quarter pair (2020 Q1 → 2020 Q2)
- Products with prices in Q1: add row with `time_dummy = 0`
- Products with prices in Q2: add row with `time_dummy = 1`
- Train one model on all pooled data

### Step 2: Price Prediction (for each product in lifecycle)

For each product that should be predicted (within its lifecycle range):

1. **Predict Q1 price**: Use model with `time_dummy = 0`
   ```python
   log_price_pred_q1 = model.predict(features, time_dummy=0)
   ```

2. **Predict Q2 price**: Use model with `time_dummy = 1`
   ```python
   log_price_pred_q2 = model.predict(features, time_dummy=1)
   ```

3. **Get actual prices** (if available):
   ```python
   log_price_actual_q1 = log(actual_price_q1)  # if exists
   log_price_actual_q2 = log(actual_price_q2)  # if exists
   ```

### Step 3: Calculate Log Deltas (for each product)

For each product in the quarter pair:

1. **Actual log delta**:
   ```python
   Log_Delta_Actual = log_price_actual_q2 - log_price_actual_q1
   ```
   - Only calculated if both Q1 and Q2 have actual prices
   - NaN if either price is missing

2. **Predicted log delta**:
   ```python
   Log_Delta_Predicted = log_price_pred_q2 - log_price_pred_q1
   ```
   - Always calculated (model predicts for all products in lifecycle)

### Step 4: Calculate Period-Level Jevons Indices

For each quarter pair (Q1 → Q2):

1. **Traditional Jevons Index**:
   ```python
   # Group all products for this quarter pair
   group = prediction_df[(prediction_df['Quarter_1'] == q1) & 
                         (prediction_df['Quarter_2'] == q2)]
   
   # Calculate mean of actual log deltas (excluding NaN)
   log_deltas_actual = group['Log_Delta_Actual'].dropna()
   mean_log_delta_traditional = np.nanmean(log_deltas_actual)
   ```

2. **Hedonic Jevons Index**:
   ```python
   # Calculate mean of predicted log deltas
   log_deltas_predicted = group['Log_Delta_Predicted'].dropna()
   mean_log_delta_hedonic = np.nanmean(log_deltas_predicted)
   ```

**Key Point**: 
- Traditional uses **actual prices** (only products with both Q1 and Q2 prices)
- Hedonic uses **predicted prices** (all products in lifecycle, even without actual prices)

### Step 5: Calculate Cumulative Indices

The cumulative indices are the **sum** of all period-level indices:

```python
# For 21 quarter pairs (2020 Q1 → 2020 Q2, ..., 2025 Q1 → 2025 Q2)
cum_traditional = sum(mean_log_delta_traditional for each quarter pair)
cum_hedonic = sum(mean_log_delta_hedonic for each quarter pair)
```

**Result**:
- Cumulative Traditional: -1.423115
- Cumulative Hedonic: -0.626765

---

## Detailed Example

### Example: Quarter Pair (2020 Q1 → 2020 Q2)

**Step 1: Training**
- Pool data: 22 products with prices in Q1 and/or Q2
- Train model: `log(price) = β₀ + β₁×features + β₂×time_dummy`
- Time dummy coefficient (β₂): -0.095 (example)

**Step 2: Prediction**
- Predict for 22 products in lifecycle range
- Q1 prediction: `log(price_q1) = β₀ + β₁×features + β₂×0`
- Q2 prediction: `log(price_q2) = β₀ + β₁×features + β₂×1`
- Difference: `log(price_q2) - log(price_q1) = β₂ = -0.095`

**Step 3: Calculate Deltas**
- For each product:
  - `Log_Delta_Actual = log(actual_price_q2) - log(actual_price_q1)` (if both exist)
  - `Log_Delta_Predicted = log(pred_price_q2) - log(pred_price_q1)` (always)

**Step 4: Period-Level Jevons**
- Traditional: `mean(Log_Delta_Actual)` = -0.095334
- Hedonic: `mean(Log_Delta_Predicted)` = -0.095122

**Step 5: Cumulative**
- Add to running total across all 21 quarter pairs

---

## Why Are They Different?

### Traditional Jevons Index (-1.423115)

- **Based on**: Actual prices
- **Products included**: Only products with actual prices in both quarters
- **No quality adjustment**: Direct price changes
- **Reflects**: Actual market price changes, including quality changes

### Hedonic Jevons Index (-0.626765)

- **Based on**: Predicted prices (from hedonic regression)
- **Products included**: All products in lifecycle range (even without actual prices)
- **Quality adjusted**: Controls for product features
- **Reflects**: Pure price changes, holding quality constant

### The Large Difference (79.64 percentage points)

The difference between Traditional (-142.31%) and Hedonic (-62.68%) is **79.64 percentage points**. This indicates:

1. **Quality improvements**: Products are getting better features over time
2. **Feature value changes**: The marginal value of features is changing
3. **Model differences**: Time Dummy model's pooled approach captures different patterns than other models

**Why Time Dummy shows larger quality adjustment effect?**
- Pooled data approach: Shares parameters across periods
- Time dummy captures average time effect across all products
- May smooth out some period-specific variations
- Different from independent period models (Basic Hedonic) or direct delta modeling (Delta Model)

---

## Comparison with Other Models

| Model | Traditional | Hedonic | Quality Adjustment Effect |
|-------|------------|---------|---------------------------|
| **Time Dummy** | -142.31% | -62.68% | **55.96%** |
| **Delta Model** | -142.31% | -141.10% | 0.85% |
| **Delta + Error** | -142.31% | -140.20% | 1.48% |
| **Basic Hedonic** | - | -67.59% | - |

**Key Observation**: Time Dummy model shows the **largest quality adjustment effect** (55.96%), much larger than Delta models (0.85-1.48%). This suggests:

1. **Different modeling approach**: Pooled data + time dummy captures different patterns
2. **Parameter sharing**: Single model for both periods may smooth variations
3. **Time effect interpretation**: Time dummy coefficient represents average time effect

---

## Mathematical Formulation

### For Each Quarter Pair (t, t+1)

**Model**: `log(price) = β₀ + Σβᵢ×featureᵢ + βₜ×time_dummy`

**Prediction**:
- Q1: `log(price_q1) = β₀ + Σβᵢ×featureᵢ + βₜ×0 = β₀ + Σβᵢ×featureᵢ`
- Q2: `log(price_q2) = β₀ + Σβᵢ×featureᵢ + βₜ×1 = β₀ + Σβᵢ×featureᵢ + βₜ`

**Predicted Delta**:
```
log(price_q2) - log(price_q1) = βₜ
```

**Jevons Index**:
- Traditional: `mean(log(actual_price_q2) - log(actual_price_q1))`
- Hedonic: `mean(log(pred_price_q2) - log(pred_price_q1)) = mean(βₜ) ≈ βₜ`

**Cumulative**:
```
Cumulative = Σ(Period-Level Jevons Index) for all 21 quarter pairs
```

---

## Code Reference

The calculation is implemented in:
- `Quarter/lasso_time_dummy_model.py`:
  - Lines 214-215: Period-level Jevons calculation
  - Lines 271-272: Log delta calculation
  - Lines 293-305: Traditional and Hedonic Jevons from deltas
  - Lines 324-325: Cumulative calculation

---

## Summary

1. **Traditional Jevons Index (-142.31%)**:
   - Sum of mean actual log price changes across 21 quarter pairs
   - Based on actual prices only
   - No quality adjustment

2. **Hedonic Jevons Index (-62.68%)**:
   - Sum of mean predicted log price changes across 21 quarter pairs
   - Based on hedonic regression predictions
   - Quality adjusted (controls for features)
   - Includes all products in lifecycle (even without actual prices)

3. **Large Difference (79.64%)**:
   - Indicates significant quality improvements over time
   - Time Dummy model's pooled approach captures different patterns
   - Much larger quality adjustment effect than Delta models

---

*Reference: `Quarter/lasso_time_dummy_model.py`*


