# Time Dummy vs Traditional Jevons Index: Calculation Comparison

## Question

Is there a difference between the "actual price difference" (Traditional Jevons Index) calculated in the Time Dummy model and the original Traditional Jevons Index?

## Answer

**The calculation formula is identical, but the product sets may differ slightly.**

---

## Calculation Formula Comparison

### Original Traditional Jevons Index

**Code**: `Quarter/quarterly_jevons_index_calculator.py`

```python
# For each quarter pair (Q1 → Q2)
valid_mask = (quarter1_prices > 0) & (quarter2_prices > 0) & \
             (~quarter1_prices.isna()) & (~quarter2_prices.isna())

q1_valid = quarter1_prices[valid_mask]
q2_valid = quarter2_prices[valid_mask]

# Calculate log price ratios
log_price_ratios = np.log(q2_valid) - np.log(q1_valid)

# Jevons Index = mean of log price ratios
jevons_index = np.mean(log_price_ratios)
```

**Formula**: 
```
Traditional_Jevons = mean(log(price_Q2) - log(price_Q1))
```
for all products with valid prices in both Q1 and Q2.

### Time Dummy Traditional Jevons Index

**Code**: `Quarter/lasso_time_dummy_model.py`

```python
# For each product in lifecycle range
for product_idx in products_to_predict:
    if (has actual prices in both Q1 and Q2):
        log_delta_actual = log(price_actual_q2) - log(price_actual_q1)
    else:
        log_delta_actual = NaN

# For each quarter pair
log_deltas_actual = group['Log_Delta_Actual'].dropna()
mean_log_delta_traditional = np.nanmean(log_deltas_actual)
```

**Formula**:
```
Traditional_Jevons = mean(log(price_Q2) - log(price_Q1))
```
for all products in lifecycle range with valid prices in both Q1 and Q2.

---

## Key Differences

### 1. Product Set

**Original Traditional**:
- Includes **all products** in the dataset
- Only requirement: valid prices in both Q1 and Q2

**Time Dummy Traditional**:
- Includes only products **within their lifecycle range**
- Lifecycle range: from entry-1 to exit+1 (or until 2025 Q2)
- Additional requirement: product must be in lifecycle range

**Example**:
- Product enters market in 2022 Q1, exits in 2024 Q4
- Original Traditional: includes this product for all quarter pairs where it has prices
- Time Dummy: includes this product only for quarter pairs within 2021 Q4 to 2025 Q1

### 2. Calculation Method

**Both use the same formula**: `mean(log(price_Q2) - log(price_Q1))`

The calculation logic is **identical** - both calculate the mean of log price differences.

### 3. Expected Results

**If product sets are identical**: Results should be **exactly the same**

**If product sets differ**: Results will differ slightly

---

## Why They Might Differ

### Scenario 1: Same Product Set

If all products in the dataset are within their lifecycle range for all relevant quarter pairs, the results should be **identical**.

### Scenario 2: Different Product Set

If some products are excluded from Time Dummy (outside lifecycle range), the results will differ:

1. **Fewer products in Time Dummy**: 
   - May have slightly different mean if excluded products have different price change patterns
   
2. **More products in Time Dummy**:
   - Unlikely, as Time Dummy is more restrictive (lifecycle requirement)

### Scenario 3: Edge Cases

- Products entering/leaving market at different times
- Products with prices outside their lifecycle range
- Missing data handling differences

---

## Actual Comparison

From the results:
- **Traditional (Original)**: -1.423115 (-142.31%)
- **Time Dummy Traditional**: -1.423115 (-142.31%)

**They are identical!**

This suggests:
1. The product sets are effectively the same
2. The calculation methods produce identical results
3. Lifecycle filtering doesn't significantly change the product set for Traditional calculation

---

## Why They Are Identical

### Reason 1: Lifecycle Range is Broad

The lifecycle range (entry-1 to exit+1) is very inclusive:
- Most products have prices throughout their lifecycle
- Entry-1 and exit+1 extensions capture most relevant periods
- Very few products are excluded

### Reason 2: Same Calculation Logic

Both use:
```python
mean(log(price_Q2) - log(price_Q1))
```

The mathematical operation is identical.

### Reason 3: Same Data Source

Both read from the same `Dataset.xlsx` file and use the same price columns.

---

## Important Discovery: Calculation Method Difference

### Two Different Calculations in Time Dummy Model

**In `run_time_dummy_models()` function** (Model Summary):
```python
# Method 1: Mean of means (INCORRECT for Jevons Index)
jevons_actual = mean(log_prices_actual_q2) - mean(log_prices_actual_q1)
```
This is: `mean(log(Q2)) - mean(log(Q1))`

**In `calculate_jevons_indices()` function** (Final Jevons Indices):
```python
# Method 2: Mean of deltas (CORRECT for Jevons Index)
mean_log_delta_traditional = mean(Log_Delta_Actual)
```
This is: `mean(log(Q2) - log(Q1))`

### Mathematical Difference

**These are NOT equivalent**:
- `mean(log(Q2) - log(Q1))` = Traditional Jevons Index (correct)
- `mean(log(Q2)) - mean(log(Q1))` = Different calculation (incorrect for Jevons)

**When are they equal?**
- Only when all products have the same price change
- In general: `mean(A - B) ≠ mean(A) - mean(B)`

### Actual Results

From the data:
- **Model Summary** (Method 1): -0.095029
- **Jevons Indices** (Method 2): -0.095334
- **Difference**: 0.000305 (small but not zero)

The **final cumulative Traditional Jevons Index** (-1.423115) uses **Method 2** (correct formula), which matches the original Traditional Jevons Index exactly.

## Conclusion

**The Traditional Jevons Index in Time Dummy model's final output uses the same calculation formula as the original Traditional Jevons Index.**

**Key Points**:
1. ✅ **Final output uses correct formula**: `mean(log(price_Q2) - log(price_Q1))`
2. ✅ **Same calculation logic**: Mean of log price differences
3. ⚠️ **Model Summary uses different method**: `mean(log_Q2) - mean(log_Q1)` (for display only)
4. ✅ **In practice**: Final results are identical to original Traditional, confirming correct calculation
5. ⚠️ **Product sets**: Time Dummy includes products in lifecycle range (may differ slightly)

**The "actual price difference" in Time Dummy's final output is calculated the same way as Traditional Jevons Index - using the correct formula `mean(log(price_Q2) - log(price_Q1))`.**

---

*Reference*:
- `Quarter/quarterly_jevons_index_calculator.py` - Original Traditional calculation
- `Quarter/lasso_time_dummy_model.py` - Time Dummy Traditional calculation

