# 模型结构总结

## 一、模型文件结构

### 季度模型（Quarter/）

1. **`lasso_price_prediction.py`** - **基础模型**
   - 功能：训练季度Lasso价格预测模型
   - 输出：`Lasso_Price_Predictions.xlsx`
   - 提供函数：
     - `preprocess_features()` - 特征预处理
     - `get_feature_columns()` - 获取9个特征列名
     - `get_trained_models()` - 返回训练好的模型和scaler

2. **`predicted_jevons_index_calculator.py`** - Hedonic Jevons指数
   - 功能：使用基础模型预测价格，计算Hedonic Jevons指数
   - 方法：内部独立训练模型（与基础模型逻辑一致）
   - 输出：`Predicted_Quarterly_Jevons_Index_Results.xlsx`

3. **`predicted_jevons_index_with_error.py`** - Hedonic Jevons指数（带Error Feature）
   - 功能：使用前一个季度的预测误差作为特征
   - 方法：调用`get_trained_models()`获取基础模型，然后添加error feature
   - 输出：`Predicted_Jevons_Index_With_Error_Results.xlsx`

4. **`lasso_delta_price_change.py`** - Delta模型
   - 功能：直接建模价格变化（log价格差值）
   - 方法：独立训练，使用OOF预测
   - 输出：`Lasso_Delta_Models1.xlsx`

5. **`lasso_delta_price_change_with_error.py`** - Delta模型（带Error Feature）
   - 功能：Delta模型 + 前一个期间对的误差特征
   - 方法：独立训练，使用OOF预测
   - 输出：`Lasso_Delta_Models_With_Error.xlsx`

### 年度模型（annual/）

1. **`lasso_price_prediction_annual.py`** - **基础模型**
   - 功能：训练年度Lasso价格预测模型
   - 方法：从季度数据聚合到年度平均值，然后训练年度模型
   - 输出：`Lasso_Price_Predictions_Annual.xlsx`
   - 提供函数：
     - `aggregate_quarters_to_years()` - 将季度数据聚合为年度数据
     - `get_trained_models()` - 返回训练好的年度模型和scaler
     - `run_annual_lasso_regression()` - 运行年度Lasso回归

2. **`predicted_annual_jevons_index_calculator.py`** - Hedonic年度Jevons指数
   - 功能：使用年度数据训练模型，预测价格，计算Hedonic Jevons指数
   - 方法：内部独立训练年度Lasso模型（与季度基础模型逻辑一致）
   - 输出：`Predicted_Annual_Jevons_Index_Results.xlsx`

2. **`predicted_annual_jevons_index_with_error.py`** - Hedonic年度Jevons指数（带Error Feature）
   - 功能：使用前一年的预测误差作为特征
   - 方法：内部训练基础模型，然后添加error feature
   - 输出：`Predicted_Annual_Jevons_Index_With_Error_Results.xlsx`

3. **`lasso_delta_price_change_annual.py`** - 年度Delta模型
   - 功能：直接建模年度价格变化
   - 方法：独立训练，使用OOF预测
   - 输出：`Lasso_Delta_Models_Annual.xlsx`

4. **`lasso_delta_price_change_annual_with_error.py`** - 年度Delta模型（带Error Feature）
   - 功能：年度Delta模型 + 前一个年度对的误差特征
   - 方法：独立训练，使用OOF预测
   - 输出：`Lasso_Delta_Models_Annual_With_Error.xlsx`

## 二、9个特征详细说明

所有模型使用相同的**9个特征**（来自 `get_feature_columns()`）：

| # | 特征名称 | 说明 | 原始数据列 | 预处理 |
|---|---------|------|-----------|--------|
| 1 | `is_ios` | 操作系统 | `Company Name` | Apple=1, 其他=0 |
| 2 | `mobile_weight_numeric` | 手机重量（克） | `Mobile Weight` | 提取数字值，默认200g |
| 3 | `ram_mem_numeric` | RAM内存（GB） | `RAM` 或 `Ram Mem` | 提取数字值，默认4GB |
| 4 | `front_camera_mp` | 前置摄像头（MP） | `Front Camera` | 提取MP值，默认8MP |
| 5 | `max_mp_numeric` | 后置摄像头最大MP | `Max_MP` | 转换为数值，默认12MP |
| 6 | `num_cameras_numeric` | 摄像头数量 | `Num_Cameras` | 转换为数值，默认2个 |
| 7 | `processor_level_encoded` | 处理器等级 | `Processor Level` | LabelEncoder编码：Entry Level, Midrange, Flagship |
| 8 | `battery_capacity_numeric` | 电池容量（mAh） | `Battery Capacity` | 提取数字值，默认3000mAh |
| 9 | `screen_size_numeric` | 屏幕尺寸（英寸） | `Screen Size` | 提取数字值，默认6.0英寸 |

### 特征预处理细节

所有特征预处理在 `preprocess_features()` 函数中完成：

1. **is_ios**: 基于公司名称判断（`Company Name == 'Apple'`）
2. **数值提取**: 使用正则表达式从字符串中提取数字（如"200g" → 200）
3. **Processor Level标准化**: 
   - 统一为三个类别：'Entry Level', 'Midrange', 'Flagship'
   - 使用关键词匹配处理变体（如"flagship", "FLAGSHIP"等）
   - 缺失值填充为'Unknown'
4. **缺失值处理**: 使用合理的默认值填充

## 三、模型训练参数

所有模型使用**完全一致**的Lasso参数：

```python
LassoCV(
    cv=min(5, len(data)//2),  # 自适应交叉验证折数
    random_state=42,           # 随机种子
    max_iter=2000              # 最大迭代次数
)
```

特征标准化：
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 四、模型差异总结

| 维度 | 季度基础模型 | 年度模型 |
|------|------------|---------|
| **基础模型文件** | ✓ 有独立文件 | ✓ 有独立文件 |
| **特征预处理** | ✓ 使用 `preprocess_features()` | ✓ 使用 `preprocess_features()` |
| **Lasso参数** | ✓ 完全一致 | ✓ 完全一致 |
| **数据粒度** | 季度数据 | 年度聚合数据（季度平均值） |
| **模型数量** | 每个季度一个模型 | 每个年度一个模型 |

## 五、为什么年度模型没有独立的基础模型文件？

年度模型的设计是：
1. **复用特征预处理**：直接使用 `Quarter/lasso_price_prediction.py` 中的函数
2. **内部训练**：在各自的脚本中训练年度模型，逻辑与季度基础模型完全一致
3. **避免重复**：不需要单独维护一个年度基础模型文件，因为训练逻辑相同

**结论**：虽然年度模型没有独立的基础模型文件，但所有模型使用相同的Lasso参数和特征预处理，确保了一致性。

