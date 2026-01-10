# 所有模型综合总结

## 一、模型概览

本项目共包含**14个模型**（7种类型 × 2个时间维度：季度/年度）：

### 季度模型（Quarter）
1. Traditional Jevons Index
2. Basic Lasso (Hedonic)
3. Basic Hedonic with Error Feature
4. Delta Model
5. Delta Model with Error Feature
6. Time Dummy Model

### 年度模型（Annual）
1. Traditional Jevons Index
2. Basic Lasso (Hedonic)
3. Basic Hedonic with Error Feature
4. Delta Model
5. Delta Model with Error Feature
6. Time Dummy Model

---

## 二、模型详细说明

### 1. Traditional Jevons Index（传统Jevons指数）

**文件**：
- 季度：`Quarter/quarterly_jevons_index_calculator.py`
- 年度：`annual/annually_jevons_index_calculator.py`

**方法**：
- 直接使用实际价格数据
- 计算相邻period的log价格差值：`log(price_t+1) - log(price_t)`
- Jevons Index = 所有产品log差值的均值

**特点**：
- ✅ 无质量调整
- ✅ 反映实际市场价格变化
- ✅ 作为基准对比

**预测范围**：
- 从进入市场前一个period到退出市场后一个period（直到2025Q2/2025）

**输出**：
- 相邻period比较
- 所有period对比较
- 同period跨年比较（仅季度）

---

### 2. Basic Lasso (Hedonic)（基础Hedonic模型）

**文件**：
- 季度：`Quarter/lasso_price_prediction.py`
- 年度：`annual/lasso_price_prediction_annual.py`

**方法**：
- 每个period独立训练Lasso模型
- 使用9个特征：iOS/Android, 重量, RAM, 前置摄像头, 后置摄像头(最大MP), 摄像头数量, 处理器等级, 电池容量, 屏幕尺寸
- 预测log价格：`log(price) = f(features)`
- 使用LassoCV自动选择alpha（正则化参数）

**特点**：
- ✅ 质量调整（通过特征控制）
- ✅ 每个period独立模型
- ✅ 生成PDF回归报告（系数、置信区间、p值）

**预测范围**：
- 从进入市场前一个period到退出市场后一个period（直到2025Q2/2025）

**输出**：
- 每个产品的预测价格
- 模型性能摘要（R², Alpha, 特征选择）
- PDF回归报告

---

### 3. Basic Hedonic with Error Feature（带误差特征的Hedonic模型）

**文件**：
- 季度：`Quarter/predicted_jevons_index_with_error.py`
- 年度：`annual/predicted_annual_jevons_index_with_error.py`

**方法**：
- 顺序训练模型：period t的模型使用period t-1的预测误差作为额外特征
- 第一个period使用基础模型（无误差特征）
- 后续period：`log(price_t) = f(features_t, error_t-1)`
- 误差 = 实际价格 - 预测价格

**特点**：
- ✅ 质量调整
- ✅ 捕获时间序列依赖
- ✅ 误差特征帮助模型学习预测偏差

**预测范围**：
- 从进入市场前一个period到退出市场后一个period（直到2025Q2/2025）

**输出**：
- 每个产品的预测价格
- Hedonic Jevons Index
- 累积价格变化

---

### 4. Delta Model（Delta模型）

**文件**：
- 季度：`Quarter/lasso_delta_price_change.py`
- 年度：`annual/lasso_delta_price_change_annual.py`

**方法**：
- 直接建模log价格差值：`log(price_t+1) - log(price_t) = f(features)`
- 每个相邻period对训练一个Lasso模型
- 使用Out-of-Fold (OOF)预测避免mean-matching artifact

**特点**：
- ✅ 直接建模价格变化
- ✅ 质量调整（通过特征控制）
- ✅ 避免累积误差
- ✅ 更准确的Hedonic Jevons Index

**预测范围**：
- 从进入市场前一个period到退出市场后一个period（直到2025Q2/2025）
- 对所有在生命周期范围内的产品预测（即使没有实际价格）

**输出**：
- Traditional Jevons Index（基于实际log差值）
- Hedonic Jevons Index（基于预测log差值）
- 质量调整效应

---

### 5. Delta Model with Error Feature（带误差特征的Delta模型）

**文件**：
- 季度：`Quarter/lasso_delta_price_change_with_error.py`
- 年度：`annual/lasso_delta_price_change_annual_with_error.py`

**方法**：
- 结合Delta模型和Error Feature
- 顺序训练：period对(t, t+1)的模型使用period对(t-1, t)的预测误差作为特征
- `log(price_t+1) - log(price_t) = f(features, error_t-1)`

**特点**：
- ✅ 直接建模价格变化
- ✅ 质量调整
- ✅ 捕获时间序列依赖
- ✅ 避免累积误差

**预测范围**：
- 从进入市场前一个period到退出市场后一个period（直到2025Q2/2025）

**输出**：
- Traditional Jevons Index
- Hedonic Jevons Index
- 质量调整效应

---

### 6. Time Dummy Model（时间虚拟变量模型）

**文件**：
- 季度：`Quarter/lasso_time_dummy_model.py`
- 年度：`annual/lasso_time_dummy_model_annual.py`

**方法**：
- 将相邻两个period的数据pool在一起
- 添加time dummy variable（0 = 第一个period，1 = 第二个period）
- 训练单个Lasso模型：`log(price) = f(features, time_dummy)`
- 用同一个模型预测两个period的价格

**特点**：
- ✅ 质量调整
- ✅ 相邻period共享模型参数
- ✅ Time dummy系数直接反映时间效应
- ✅ 更高效（一个模型预测两个period）

**预测范围**：
- 从进入市场前一个period到退出市场后一个period（直到2025Q2/2025）

**输出**：
- Traditional Jevons Index（基于实际价格差值）
- Hedonic Jevons Index（基于预测价格差值）
- Time dummy系数（反映时间效应）

---

## 三、模型对比表

| 模型类型 | 质量调整 | 时间依赖 | 建模方式 | 主要特点 |
|---------|---------|---------|---------|---------|
| **Traditional** | ❌ | ❌ | 直接计算 | 基准，无质量调整 |
| **Basic Hedonic** | ✅ | ❌ | 独立period模型 | 标准hedonic回归 |
| **Basic + Error** | ✅ | ✅ | 顺序训练，误差特征 | 捕获时间依赖 |
| **Delta** | ✅ | ❌ | 直接建模价格变化 | 避免累积误差 |
| **Delta + Error** | ✅ | ✅ | Delta + 误差特征 | 结合两者优势 |
| **Time Dummy** | ✅ | ✅ | Pooled data + time dummy | 共享参数，高效 |

---

## 四、关键结果总结

### 季度模型累积结果

| 模型 | Traditional累积 | Hedonic累积 | 质量调整效应 |
|------|----------------|------------|-------------|
| Traditional | -1.423 (-142.31%) | - | - |
| Basic Hedonic | - | -0.676 (-67.59%) | - |
| Basic + Error | - | -0.723 (-72.26%) | - |
| Delta | -1.423 (-142.31%) | -1.411 (-141.10%) | 0.85% |
| Delta + Error | -1.423 (-142.31%) | -1.402 (-140.20%) | 1.48% |
| Time Dummy | -1.423 (-142.31%) | -0.627 (-62.68%) | 55.96% |

### 年度模型累积结果

| 模型 | Traditional累积 | Hedonic累积 | 质量调整效应 |
|------|----------------|------------|-------------|
| Traditional | -1.214 (-121.35%) | - | - |
| Basic Hedonic | - | -0.739 (-73.87%) | - |
| Basic + Error | - | -0.732 (-73.25%) | - |
| Delta | -1.214 (-121.35%) | -1.197 (-119.70%) | 1.36% |
| Delta + Error | -1.214 (-121.35%) | -1.197 (-119.68%) | 1.38% |
| Time Dummy | -1.214 (-121.35%) | -0.777 (-77.69%) | 35.98% |

---

## 五、模型选择建议

### 1. 基准对比
- **Traditional Jevons Index**：作为无质量调整的基准

### 2. 标准Hedonic分析
- **Basic Hedonic**：标准方法，每个period独立模型
- **Time Dummy**：更高效，相邻period共享参数

### 3. 时间序列分析
- **Basic + Error**：捕获时间依赖，适合序列数据
- **Delta + Error**：结合价格变化建模和时间依赖

### 4. 最准确的质量调整
- **Delta Model**：直接建模价格变化，避免累积误差
- **Delta + Error**：在Delta基础上加入时间依赖

---

## 六、技术细节

### 共同特征
- **9个特征**：iOS/Android, 重量, RAM, 前置摄像头, 后置摄像头(最大MP), 摄像头数量, 处理器等级, 电池容量, 屏幕尺寸
- **Lasso回归**：L1正则化，自动特征选择
- **LassoCV**：交叉验证选择alpha
- **StandardScaler**：特征标准化
- **预测范围**：从进入市场前一个period到退出市场后一个period

### 模型参数
- `random_state=42`：确保可重复性
- `max_iter=2000`：最大迭代次数
- `cv=min(5, n_samples//2)`：交叉验证折数

---

## 七、输出文件

### 季度模型输出
1. `Quarterly_Jevons_Index_Results.xlsx` - Traditional Jevons
2. `Lasso_Price_Predictions.xlsx` - Basic Hedonic
3. `Lasso_Regression_Summary.pdf` - Basic Hedonic回归报告
4. `Predicted_Quarterly_Jevons_Index_Results.xlsx` - Basic Hedonic Jevons
5. `Predicted_Jevons_Index_With_Error_Results.xlsx` - Basic + Error
6. `Lasso_Delta_Models1.xlsx` - Delta Model
7. `Lasso_Delta_Models_With_Error.xlsx` - Delta + Error
8. `Lasso_Time_Dummy_Models.xlsx` - Time Dummy

### 年度模型输出
1. `Annual_Jevons_Index_Results.xlsx` - Traditional Jevons
2. `Lasso_Price_Predictions_Annual.xlsx` - Basic Hedonic
3. `Lasso_Regression_Summary_Annual.pdf` - Basic Hedonic回归报告
4. `Predicted_Annual_Jevons_Index_Results.xlsx` - Basic Hedonic Jevons
5. `Predicted_Annual_Jevons_Index_With_Error_Results.xlsx` - Basic + Error
6. `Lasso_Delta_Models_Annual.xlsx` - Delta Model
7. `Lasso_Delta_Models_Annual_With_Error.xlsx` - Delta + Error
8. `Lasso_Time_Dummy_Models_Annual.xlsx` - Time Dummy

---

## 八、关键发现

1. **质量调整效应**：
   - Delta模型的质量调整效应较小（0.85-1.48%）
   - Time Dummy模型的质量调整效应较大（35.98-55.96%）
   - 说明不同模型对质量调整的敏感度不同

2. **时间依赖**：
   - Error Feature模型捕获了时间序列依赖
   - Delta + Error模型结合了价格变化建模和时间依赖

3. **模型准确性**：
   - Delta模型直接建模价格变化，避免累积误差
   - Time Dummy模型通过pooled data提高效率

4. **累积价格变化**：
   - Traditional：-142.31%（季度），-121.35%（年度）
   - Hedonic（Delta）：-141.10%（季度），-119.70%（年度）
   - 质量调整后，价格下降幅度略有减小

---

*最后更新：2025年10月*

