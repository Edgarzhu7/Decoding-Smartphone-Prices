# 研究结果总结 (Research Results Summary)

## 一、数据概况 (Data Overview)

- **数据集**: `Dataset.xlsx`
- **产品数量**: 152款手机（有价格数据的模型）
- **时间跨度**: 2018 Q4 - 2025 Q3（季度数据）
- **特征变量**: 9个特征（操作系统、重量、RAM、前后摄像头、处理器等级、电池容量、屏幕尺寸）

---

## 二、季度分析结果 (Quarterly Analysis Results)

### 2.1 传统Jevons指数 (Traditional Jevons Index)
**文件**: `Quarter/Quarterly_Jevons_Index_Results.xlsx`

- **分析期间**: 2020 Q1 → 2025 Q3（27个相邻季度对）
- **累积Jevons指数**: **-1.7645**
- **累积价格变化**: **-176.45%**（相当于价格下降约82.8%）
- **解释**: 基于实际市场价格的几何平均，未考虑质量调整

### 2.2 Hedonic Jevons指数（基于预测价格）
**文件**: `Quarter/Predicted_Quarterly_Jevons_Index_Results.xlsx`

- **分析期间**: 2020 Q1 → 2025 Q3（22个相邻季度对）
- **累积Jevons指数**: **-0.6914**
- **累积价格变化**: **-69.14%**（相当于价格下降约50.0%）
- **解释**: 基于Lasso模型预测的质量调整价格

### 2.3 Hedonic Jevons指数（带Error Feature）
**文件**: `Quarter/Predicted_Jevons_Index_With_Error_Results.xlsx`

- **累积Jevons指数**: **-0.7524**
- **累积价格变化**: **-75.24%**
- **特点**: 使用前一个季度的预测误差作为特征，捕捉时间序列依赖性
- **改进**: 相比基础Hedonic模型，R²有所提升

### 2.4 Delta模型（直接拟合价格变化）
**文件**: `Quarter/Lasso_Delta_Models1.xlsx`

- **传统累积**: **-1.4377**
- **Hedonic累积**: **-1.4132**
- **差异**: **0.0244**（仅2.44%）
- **解释**: 直接建模相邻季度间的log价格差值，使用Out-of-Fold预测避免过拟合
- **优势**: 传统和Hedonic指数差异很小，说明直接建模价格变化能更好地捕捉特征效应

### 2.5 Lasso模型性能
**文件**: `Quarter/Lasso_Price_Predictions.xlsx`

- **平均R²**: **0.5746**（57.46%的解释力）
- **样本数**: 从2020 Q1的约50个样本增长到2025 Q3的133个样本
- **特征选择**: 通常选择4-9个特征（共9个候选特征）
- **Alpha值**: 随时间递减（从0.04降至0.0002），说明模型复杂度增加

---

## 三、年度分析结果 (Annual Analysis Results)

### 3.1 传统年度Jevons指数
**文件**: `annual/Annual_Jevons_Index_Results.xlsx`

- **分析期间**: 2018 → 2025（7个相邻年度对）
- **累积Jevons指数**: **-1.6003**
- **累积价格变化**: **-160.03%**（相当于价格下降约79.8%）
- **年度变化**:
  - 2018→2019: -3.13%
  - 2019→2020: -26.84%
  - 2020→2021: -27.58%
  - 2021→2022: -31.75%
  - 2022→2023: -24.83%
  - 2023→2024: -23.05%
  - 2024→2025: -22.85%

### 3.2 Hedonic年度Jevons指数
**文件**: `annual/Predicted_Annual_Jevons_Index_Results.xlsx`

- **分析期间**: 2020 → 2025（5个相邻年度对）
- **累积Jevons指数**: **-0.8220**
- **累积价格变化**: **-82.20%**（相当于价格下降约56.0%）
- **年度变化**:
  - 2020→2021: -11.98%
  - 2021→2022: -16.34%
  - 2022→2023: -16.93%
  - 2023→2024: -16.31%
  - 2024→2025: -20.64%

### 3.3 Hedonic年度Jevons指数（带Error Feature）
**文件**: `annual/Predicted_Annual_Jevons_Index_With_Error_Results.xlsx`

- **特点**: 使用前一年的预测误差作为特征
- **用途**: 捕捉年度间的自相关效应

---

## 四、关键发现 (Key Findings)

### 4.1 价格指数差异

| 方法 | 累积Jevons指数 | 价格下降幅度 | 质量调整效应 |
|------|---------------|-------------|-------------|
| **季度传统** | -1.7645 | 82.8% | - |
| **季度Hedonic** | -0.6914 | 50.0% | **32.8个百分点** |
| **年度传统** | -1.6003 | 79.8% | - |
| **年度Hedonic** | -0.8220 | 56.0% | **23.8个百分点** |
| **Delta传统** | -1.4377 | 76.2% | - |
| **Delta Hedonic** | -1.4132 | 75.7% | **0.5个百分点** |

### 4.2 主要结论

1. **传统指数高估通缩**: 
   - 季度层面：传统指数显示82.8%的价格下降，而质量调整后为50.0%
   - **差异约32.8个百分点**可归因于质量改进而非纯价格下降

2. **Delta模型表现更一致**:
   - 直接建模价格变化时，传统和Hedonic指数差异仅0.5个百分点
   - 说明直接建模能更好地捕捉特征对价格变化的影响

3. **年度vs季度**:
   - 年度数据平滑了季度波动
   - 年度Hedonic指数（56.0%下降）介于季度传统（82.8%）和季度Hedonic（50.0%）之间

4. **模型性能**:
   - Lasso模型平均R² = 57.46%，说明特征能解释约一半的价格变异
   - 样本数随时间增长，模型稳定性提升
   - Error Feature模型在某些季度表现更好

---

## 五、文件结构 (File Structure)

### 季度分析 (Quarter/)
- `quarterly_jevons_index_calculator.py` - 传统季度Jevons指数计算
- `predicted_jevons_index_calculator.py` - Hedonic季度Jevons指数计算
- `predicted_jevons_index_with_error.py` - 带Error Feature的Hedonic指数
- `lasso_delta_price_change.py` - Delta模型（直接建模价格变化）
- `lasso_price_prediction.py` - Lasso价格预测模型

### 年度分析 (annual/)
- `annually_jevons_index_calculator.py` - 传统年度Jevons指数计算
- `predicted_annual_jevons_index_calculator.py` - Hedonic年度Jevons指数计算
- `predicted_annual_jevons_index_with_error.py` - 带Error Feature的年度Hedonic指数

### 数据文件
- `Dataset.xlsx` - 原始数据集（152款手机，2018 Q4 - 2025 Q3）
- 所有结果Excel文件包含多个工作表：模型摘要、系数、预测价格、Jevons指数等

---

## 六、方法学说明 (Methodology Notes)

1. **Jevons指数计算**: 使用log价格差值的均值（不使用exp），直接反映价格变化百分比
2. **特征预处理**: 统一使用`lasso_price_prediction.py`中的预处理函数
3. **生命周期预测**: 仅在产品有实际价格的期间进行预测
4. **交叉验证**: Lasso使用5折交叉验证选择最优alpha
5. **Out-of-Fold预测**: Delta模型使用OOF预测避免过拟合

---

## 七、研究意义 (Research Implications)

1. **政策含义**: 传统价格指数可能高估了通缩程度，需要质量调整
2. **消费者福利**: 质量改进带来的价值提升应计入福利计算
3. **方法创新**: Delta模型提供了更一致的价格变化建模方法
4. **特征价值**: 不同硬件特征的价值随时间呈现异质性变化（非均匀下降）

---

*最后更新: 2025年*
*数据范围: 2018 Q4 - 2025 Q3*
*产品数量: 152款手机*

