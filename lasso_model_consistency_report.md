# Lasso模型一致性检查报告

## 检查结果：✓ 所有模型使用一致的Lasso设置

### 一、LassoCV参数一致性

所有模型（季度和年度）都使用以下参数：

1. **random_state = 42** ✓
   - 所有模型统一使用42作为随机种子，确保结果可复现

2. **max_iter = 2000** ✓
   - 所有模型统一使用2000作为最大迭代次数

3. **cv = min(5, len(data)//2)** ✓
   - 所有模型使用自适应交叉验证折数
   - 根据数据量自动调整，但最多5折
   - 对于小样本，使用更少的折数

### 二、特征预处理一致性

所有模型都使用：

1. **preprocess_features()** ✓
   - 统一从 `lasso_price_prediction.py` 导入
   - 确保所有模型使用相同的特征预处理逻辑

2. **get_feature_columns()** ✓
   - 统一从 `lasso_price_prediction.py` 导入
   - 确保所有模型使用相同的**9个特征**：
     1. `is_ios` - 操作系统（iOS=1, Android=0）
     2. `mobile_weight_numeric` - 手机重量（克）
     3. `ram_mem_numeric` - RAM内存（GB）
     4. `front_camera_mp` - 前置摄像头（MP）
     5. `max_mp_numeric` - 后置摄像头最大MP
     6. `num_cameras_numeric` - 摄像头数量
     7. `processor_level_encoded` - 处理器等级（编码后的分类变量：Entry Level, Midrange, Flagship）
     8. `battery_capacity_numeric` - 电池容量（mAh）
     9. `screen_size_numeric` - 屏幕尺寸（英寸）

3. **StandardScaler()** ✓
   - 所有模型在训练前都进行特征标准化
   - 确保特征在相同尺度上

### 三、模型列表

#### 季度模型（Quarter/）
1. `lasso_price_prediction.py` - 基础Lasso价格预测模型
2. `predicted_jevons_index_calculator.py` - Hedonic Jevons指数计算
3. `predicted_jevons_index_with_error.py` - Hedonic Jevons指数（带Error Feature）
4. `lasso_delta_price_change.py` - Delta模型
5. `lasso_delta_price_change_with_error.py` - Delta模型（带Error Feature）

#### 年度模型（annual/）
1. **基础模型**：年度模型没有独立的基础模型文件，它们都：
   - 使用 `Quarter/lasso_price_prediction.py` 中的 `preprocess_features()` 和 `get_feature_columns()`
   - 在各自文件中内部训练Lasso模型（与季度基础模型使用相同的参数和逻辑）
2. `predicted_annual_jevons_index_calculator.py` - Hedonic年度Jevons指数（内部训练模型）
3. `predicted_annual_jevons_index_with_error.py` - Hedonic年度Jevons指数（带Error Feature，内部训练模型）
4. `lasso_delta_price_change_annual.py` - 年度Delta模型
5. `lasso_delta_price_change_annual_with_error.py` - 年度Delta模型（带Error Feature）

**注意**：年度模型虽然没有独立的基础模型文件，但所有模型都使用相同的Lasso参数和特征预处理逻辑，与季度基础模型完全一致。

### 四、模型差异说明

虽然所有模型使用相同的Lasso参数和特征预处理，但它们在以下方面有差异：

1. **目标变量**：
   - 传统Hedonic模型：`ln(P_t)` - 绝对价格
   - Delta模型：`ln(P_{t+1}) - ln(P_t)` - 价格变化

2. **特征数量**：
   - 基础模型：9个特征
   - Error Feature模型：10个特征（9个基础特征 + 前一期误差）

3. **训练数据**：
   - 季度模型：使用季度数据
   - 年度模型：使用年度聚合数据（季度平均值）

4. **预测方法**：
   - 传统模型：直接预测价格
   - Delta模型：直接预测价格变化，使用OOF预测避免过拟合

### 五、结论

✅ **所有模型使用完全一致的Lasso设置**，包括：
- 相同的随机种子（random_state=42）
- 相同的最大迭代次数（max_iter=2000）
- 相同的交叉验证策略（自适应CV）
- 相同的特征预处理函数
- 相同的特征标准化方法

这确保了：
1. **结果可复现性**：相同的随机种子保证结果可复现
2. **方法一致性**：所有模型使用相同的特征工程和标准化
3. **公平比较**：不同模型之间的差异仅来自方法本身，而非参数设置

---

*检查日期：2025年*
*检查工具：check_lasso_consistency_detailed.py*

