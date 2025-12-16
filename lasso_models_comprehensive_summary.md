# Lasso模型在项目中的全面应用总结

## 一、项目概述

本项目使用**Lasso回归**（Least Absolute Shrinkage and Selection Operator）进行智能手机价格的Hedonic回归分析，计算质量调整的Jevons价格指数。

### 1.1 研究目标

1. **预测价格**：基于手机特征预测价格
2. **质量调整**：计算Hedonic Jevons指数（质量调整后的价格指数）
3. **方法比较**：比较不同建模方法的效果
4. **时间序列分析**：分析价格变化的时间模式

### 1.2 数据概况

- **产品数量**：152款手机（有价格数据的模型）
- **时间跨度**：2018 Q4 - 2025 Q2（季度数据）
- **特征数量**：9个特征（操作系统、重量、RAM、摄像头、处理器、电池、屏幕）

---

## 二、模型架构总览

### 2.1 模型分类

项目包含**10个Lasso模型**，分为两个维度：

#### 维度1：时间粒度
- **季度模型**（Quarter/）：使用季度数据
- **年度模型**（annual/）：使用年度聚合数据（季度平均值）

#### 维度2：建模方法
1. **Basic模型**：基础Lasso价格预测
2. **Basic with Error Feature**：基础模型 + 前一期误差特征
3. **Delta模型**：直接建模价格变化
4. **Delta with Error Feature**：Delta模型 + 前一期误差特征

### 2.2 模型矩阵

| 方法 | 季度 | 年度 |
|------|------|------|
| **Basic** | ✓ | ✓ |
| **Basic + Error** | ✓ | ✓ |
| **Delta** | ✓ | ✓ |
| **Delta + Error** | ✓ | ✓ |

---

## 三、模型详细说明

### 3.1 Basic模型（基础Lasso价格预测）

#### 3.1.1 季度Basic模型
**文件**：`Quarter/lasso_price_prediction.py`

**方法**：
- 目标变量：`ln(P_t)` - 每个季度的log价格
- 特征：9个手机特征
- 训练：每个季度独立训练一个Lasso模型
- 预测：预测所有152款手机在每个季度的价格

**输出**：
- `Lasso_Price_Predictions.xlsx`
- 包含：预测价格、模型摘要、特征重要性

**特点**：
- 提供 `get_trained_models()` 函数供其他脚本使用
- 每个季度一个模型，共约22个模型（2020 Q1 - 2025 Q2）

#### 3.1.2 年度Basic模型
**文件**：`annual/lasso_price_prediction_annual.py`

**方法**：
- 数据：从季度数据聚合为年度平均值
- 目标变量：`ln(P_year)` - 每个年度的log价格
- 特征：相同的9个特征
- 训练：每个年度独立训练一个Lasso模型

**输出**：
- `Lasso_Price_Predictions_Annual.xlsx`
- 包含：预测价格、模型摘要、特征重要性

**特点**：
- 提供 `get_trained_models()` 函数
- 每个年度一个模型，共6个模型（2020-2025）

---

### 3.2 Basic with Error Feature模型

#### 3.2.1 季度Basic + Error模型
**文件**：`Quarter/predicted_jevons_index_with_error.py`

**方法**：
- **第一季度**：使用基础模型（9个特征）
- **后续季度**：添加前一个季度的预测误差作为第10个特征
- 误差计算：`error = ln(actual_price) - ln(predicted_price)`
- 顺序训练：先预测季度t，计算误差，然后用误差训练季度t+1的模型

**特点**：
- 捕捉时间序列依赖性
- 误差特征捕捉模型未解释的部分
- 某些季度R²有所提升

**输出**：
- `Predicted_Jevons_Index_With_Error_Results.xlsx`

#### 3.2.2 年度Basic + Error模型
**文件**：`annual/predicted_annual_jevons_index_with_error.py`

**方法**：
- 与季度版本相同，但使用年度数据
- 第一年使用基础模型，后续年份添加前一年的误差特征

**输出**：
- `Predicted_Annual_Jevons_Index_With_Error_Results.xlsx`

---

### 3.3 Delta模型（直接建模价格变化）

#### 3.3.1 季度Delta模型
**文件**：`Quarter/lasso_delta_price_change.py`

**方法**：
- **目标变量**：`ln(P_{t+1}) - ln(P_t)` - 相邻季度间的log价格差值
- **特征**：相同的9个特征
- **训练**：每个相邻季度对训练一个模型
- **OOF预测**：使用嵌套交叉验证生成Out-of-Fold预测

**关键特点**：
- **直接建模变化**：避免绝对价格预测的误差累积
- **OOF预测**：避免Mean-Matching Artifact
- **嵌套CV**：外层CV用于OOF预测，内层CV选择alpha

**输出**：
- `Lasso_Delta_Models1.xlsx`
- 包含：模型摘要、系数、每个产品的价格变化、传统vs Hedonic Jevons对比

**结果**：
- 传统累积：-1.4231
- Hedonic累积：-1.3987
- **差异仅2.44%**（相比Basic模型的32.8%差异大幅缩小）

#### 3.3.2 年度Delta模型
**文件**：`annual/lasso_delta_price_change_annual.py`

**方法**：
- 目标变量：`ln(P_{year+1}) - ln(P_year)` - 相邻年度间的log价格差值
- 数据：年度聚合数据
- 其他：与季度Delta模型相同

**输出**：
- `Lasso_Delta_Models_Annual.xlsx`

**结果**：
- 传统累积：-1.2135
- Hedonic累积：-1.2167
- **差异仅-0.32%**（几乎完全一致）

---

### 3.4 Delta with Error Feature模型

#### 3.4.1 季度Delta + Error模型
**文件**：`Quarter/lasso_delta_price_change_with_error.py`

**方法**：
- **第一个季度对**：使用基础Delta模型（9个特征）
- **后续季度对**：添加前一个季度对的预测误差作为第10个特征
- 误差计算：`error = actual_delta - predicted_delta`
- 顺序训练：先预测季度对(t, t+1)，计算误差，然后用误差训练季度对(t+1, t+2)的模型

**特点**：
- 结合Delta模型的优势 + Error Feature的时间依赖性
- 进一步缩小传统和Hedonic指数的差异

**输出**：
- `Lasso_Delta_Models_With_Error.xlsx`

**结果**：
- 传统累积：-1.4231
- Hedonic累积：-1.4133
- **差异仅0.99%**（比基础Delta模型的2.44%进一步缩小）

#### 3.4.2 年度Delta + Error模型
**文件**：`annual/lasso_delta_price_change_annual_with_error.py`

**方法**：
- 与季度版本相同，但使用年度数据
- 第一年对使用基础Delta模型，后续年份对添加前一年对的误差特征

**输出**：
- `Lasso_Delta_Models_Annual_With_Error.xlsx`

**结果**：
- 传统累积：-1.2135
- Hedonic累积：-1.2181
- **差异仅-0.46%**（几乎完全一致）

---

## 四、模型对比总结

### 4.1 季度模型结果对比

| 模型 | 传统Jevons | Hedonic Jevons | 差异 | 质量调整效应 |
|------|-----------|----------------|------|-------------|
| **Traditional** | -1.7499 | - | - | - |
| **Basic Hedonic** | - | -0.6768 | 32.8% | 大 |
| **Basic + Error** | - | -0.7378 | 30.0% | 大 |
| **Delta** | -1.4231 | -1.3987 | **2.44%** | 小 |
| **Delta + Error** | -1.4231 | -1.4133 | **0.99%** | 很小 |

### 4.2 年度模型结果对比

| 模型 | 传统Jevons | Hedonic Jevons | 差异 | 质量调整效应 |
|------|-----------|----------------|------|-------------|
| **Traditional** | -1.5132 | - | - | - |
| **Basic Hedonic** | - | -0.7403 | 23.8% | 大 |
| **Basic + Error** | - | -0.7414 | 23.7% | 大 |
| **Delta** | -1.2135 | -1.2167 | **-0.32%** | 极小 |
| **Delta + Error** | -1.2135 | -1.2181 | **-0.46%** | 极小 |

### 4.3 关键发现

1. **Delta模型显著缩小差异**：
   - Basic模型：差异30-33%
   - Delta模型：差异<3%
   - 说明直接建模价格变化能更好地捕捉特征效应

2. **Error Feature的改进**：
   - 在Delta模型中，Error Feature进一步缩小差异（从2.44%到0.99%）
   - 在Basic模型中，Error Feature影响较小

3. **年度vs季度**：
   - 年度数据平滑了季度波动
   - 年度Delta模型的差异更小（-0.32% vs 2.44%）

---

## 五、模型技术细节

### 5.1 统一的Lasso参数

所有模型使用**完全一致**的参数：

```python
LassoCV(
    cv=min(5, len(data)//2),  # 自适应交叉验证折数
    random_state=42,           # 随机种子
    max_iter=2000              # 最大迭代次数
)
```

### 5.2 统一的特征预处理

所有模型使用：
- `preprocess_features()` - 特征预处理函数
- `get_feature_columns()` - 获取9个特征列名
- `StandardScaler()` - 特征标准化

### 5.3 9个特征

1. `is_ios` - 操作系统（iOS=1, Android=0）
2. `mobile_weight_numeric` - 手机重量（克）
3. `ram_mem_numeric` - RAM内存（GB）
4. `front_camera_mp` - 前置摄像头（MP）
5. `max_mp_numeric` - 后置摄像头最大MP
6. `num_cameras_numeric` - 摄像头数量
7. `processor_level_encoded` - 处理器等级（编码）
8. `battery_capacity_numeric` - 电池容量（mAh）
9. `screen_size_numeric` - 屏幕尺寸（英寸）

### 5.4 超参数选择

- **方法**：LassoCV自动选择alpha
- **搜索范围**：默认100个alpha值（对数空间）
- **选择标准**：交叉验证平均得分最高
- **实际范围**：季度模型 0.0002-0.23，年度模型 0.002-0.02

---

## 六、模型应用场景

### 6.1 Basic模型

**适用场景**：
- 需要预测绝对价格水平
- 需要跨多个时期比较价格
- 需要分析特征对绝对价格的影响

**优势**：
- 直观：直接预测价格
- 灵活：可以预测任意时期的价格

**局限性**：
- 可能累积误差
- 传统和Hedonic指数差异较大（30-33%）

### 6.2 Basic + Error Feature模型

**适用场景**：
- 需要捕捉时间序列依赖性
- 需要改进模型预测精度
- 价格变化有自相关模式

**优势**：
- 捕捉未解释的时间模式
- 某些季度R²有所提升

**局限性**：
- 需要顺序训练和预测
- 对第一个时期无法使用error feature

### 6.3 Delta模型

**适用场景**：
- 关注价格变化而非绝对水平
- 需要更一致的传统和Hedonic指数
- 特征对变化的影响更稳定

**优势**：
- **显著缩小差异**：从30%降至2-3%
- 避免误差累积
- 更准确地捕捉特征对价格变化的影响

**局限性**：
- 需要连续两个时期都有数据
- 不能直接预测绝对价格

### 6.4 Delta + Error Feature模型

**适用场景**：
- 需要Delta模型的优势 + 时间依赖性
- 需要进一步缩小差异
- 价格变化有自相关模式

**优势**：
- 结合Delta和Error Feature的优势
- 差异进一步缩小（从2.44%到0.99%）

**局限性**：
- 实现更复杂
- 需要顺序训练

---

## 七、模型性能对比

### 7.1 R²得分

**季度Basic模型**：
- 平均R² ≈ 0.57（57%的解释力）
- 范围：0.4 - 0.6
- 随时间样本数增加，R²相对稳定

**年度Basic模型**：
- 平均R² ≈ 0.63（63%的解释力）
- 范围：0.53 - 0.87
- 年度数据更稳定，R²通常更高

**Delta模型**：
- R²通常较低（因为建模变化比绝对价格更难）
- 但差异更小，说明更好地捕捉了特征效应

### 7.2 特征选择

- **小样本**（20-30个）：选择2-4个特征（保守）
- **中样本**（40-80个）：选择6-8个特征（平衡）
- **大样本**（100+个）：选择8-9个特征（宽松）

### 7.3 Alpha值趋势

- **早期**（样本少）：alpha ≈ 0.01-0.04（保守）
- **后期**（样本多）：alpha ≈ 0.001-0.005（宽松）
- **趋势**：样本数越多，alpha越小（允许更多特征）

---

## 八、方法学贡献

### 8.1 Delta模型的创新

1. **直接建模变化**：避免了绝对价格预测的误差累积
2. **显著缩小差异**：传统和Hedonic指数差异从30%降至2-3%
3. **更准确的捕捉**：更好地捕捉特征对价格变化的影响

### 8.2 OOF预测的必要性

1. **避免伪影**：打破Mean-Matching Artifact
2. **真实评估**：提供无偏的预测性能估计
3. **公平比较**：允许比较传统和Hedonic指数

### 8.3 Error Feature的作用

1. **时间依赖性**：捕捉未解释的时间模式
2. **改进预测**：某些情况下提升R²
3. **进一步缩小差异**：在Delta模型中效果更明显

---

## 九、文件结构

### 9.1 季度模型文件（Quarter/）

1. `lasso_price_prediction.py` - 基础模型
2. `predicted_jevons_index_calculator.py` - Basic Hedonic
3. `predicted_jevons_index_with_error.py` - Basic + Error
4. `lasso_delta_price_change.py` - Delta
5. `lasso_delta_price_change_with_error.py` - Delta + Error

### 9.2 年度模型文件（annual/）

1. `lasso_price_prediction_annual.py` - 基础模型
2. `predicted_annual_jevons_index_calculator.py` - Basic Hedonic
3. `predicted_annual_jevons_index_with_error.py` - Basic + Error
4. `lasso_delta_price_change_annual.py` - Delta
5. `lasso_delta_price_change_annual_with_error.py` - Delta + Error

### 9.3 输出文件

每个模型都生成对应的Excel文件，包含：
- 预测结果
- 模型摘要（R²、Alpha、特征数）
- 特征重要性/系数
- Jevons指数计算结果

---

## 十、研究结论

### 10.1 主要发现

1. **传统指数高估通缩**：
   - 季度：传统显示82.8%下降，Hedonic显示50.0%下降
   - 差异约32.8个百分点可归因于质量改进

2. **Delta模型更一致**：
   - 传统和Hedonic指数差异仅2-3%
   - 说明直接建模价格变化能更好地捕捉特征效应

3. **Error Feature的改进**：
   - 在Delta模型中进一步缩小差异
   - 在Basic模型中影响较小

### 10.2 方法学意义

1. **Delta方法**：提供了更一致的价格指数估计方法
2. **OOF预测**：确保了预测的客观性和无偏性
3. **Error Feature**：捕捉了时间序列依赖性

### 10.3 政策含义

1. **质量调整的重要性**：传统指数可能高估通缩30-33个百分点
2. **消费者福利**：质量改进带来的价值应计入福利计算
3. **方法选择**：Delta方法提供了更稳健的估计

---

## 十一、模型选择建议

### 11.1 如果关注绝对价格水平

→ 使用 **Basic模型** 或 **Basic + Error模型**

### 11.2 如果关注价格变化和一致性

→ 使用 **Delta模型** 或 **Delta + Error模型**

### 11.3 如果需要时间依赖性

→ 使用 **+ Error Feature** 版本

### 11.4 如果需要更平滑的结果

→ 使用 **年度模型**

### 11.5 如果需要更细粒度分析

→ 使用 **季度模型**

---

## 十二、技术实现要点

### 12.1 数据预处理

- 统一使用 `preprocess_features()` 函数
- 处理缺失值、提取数值、编码分类变量
- 确保所有模型使用相同的特征定义

### 12.2 模型训练

- 统一使用 `LassoCV` 自动选择alpha
- 统一使用 `StandardScaler` 标准化特征
- 统一使用 `random_state=42` 确保可复现

### 12.3 预测方法

- Basic模型：直接预测
- Delta模型：使用OOF预测避免伪影
- Error Feature模型：顺序训练和预测

### 12.4 结果输出

- 所有模型输出Excel文件
- 包含预测结果、模型摘要、特征重要性
- 便于后续分析和比较

---

## 十三、总结

### 13.1 模型数量

- **总计**：10个Lasso模型
- **季度**：5个模型
- **年度**：5个模型

### 13.2 核心方法

1. **Basic模型**：预测绝对价格
2. **Delta模型**：直接建模价格变化（更一致）
3. **Error Feature**：捕捉时间依赖性
4. **OOF预测**：避免伪影，真实评估

### 13.3 关键发现

1. 传统指数高估通缩约30-33个百分点
2. Delta模型显著缩小差异（至2-3%）
3. Error Feature在Delta模型中效果更明显
4. 所有模型使用一致的参数和预处理

### 13.4 研究价值

1. **方法学贡献**：Delta方法和OOF预测的应用
2. **实证发现**：质量调整的重要性
3. **政策含义**：价格指数需要质量调整
4. **技术实现**：统一的、可复现的模型框架

---

*文档创建日期：2025年*
*项目范围：2018 Q4 - 2025 Q2*
*模型总数：10个Lasso模型*

