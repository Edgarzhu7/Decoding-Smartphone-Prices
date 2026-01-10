# Time Dummy模型中的Traditional和Hedonic Jevons Index解释

## 一、为什么Time Dummy模型会有两个Jevons Index？

Time Dummy模型虽然使用了pooled data和time dummy variable，但它仍然是一个**hedonic回归模型**，因此可以计算两个Jevons Index：

1. **Traditional Jevons Index**：基于**实际价格**的log差值
2. **Hedonic Jevons Index**：基于**预测价格**的log差值（质量调整后）

---

## 二、计算逻辑

### 2.1 模型训练

Time Dummy模型：
- 将相邻两个period（季度/年度）的数据**pool在一起**
- 添加**time dummy variable**（0 = 第一个period，1 = 第二个period）
- 训练**单个Lasso模型**来预测log价格

### 2.2 价格预测

对于每个产品，模型预测两个period的价格：
- **Q1的预测价格**：`log_price_pred_q1 = model.predict(features, time_dummy=0)`
- **Q2的预测价格**：`log_price_pred_q2 = model.predict(features, time_dummy=1)`

### 2.3 Jevons Index计算

对于每个季度对（Q1 → Q2）：

**Traditional Jevons Index**：
```python
# 使用实际价格的log差值
log_delta_actual = log(price_actual_q2) - log(price_actual_q1)
jevons_traditional = mean(log_delta_actual)
```

**Hedonic Jevons Index**：
```python
# 使用预测价格的log差值（质量调整后）
log_delta_predicted = log(price_pred_q2) - log(price_pred_q1)
jevons_hedonic = mean(log_delta_predicted)
```

---

## 三、为什么需要两个Index？

### 3.1 Traditional Jevons Index的作用

- **基准对比**：提供没有质量调整的原始价格变化
- **评估质量调整效果**：通过对比Traditional和Hedonic，可以看出质量调整的影响

### 3.2 Hedonic Jevons Index的作用

- **质量调整**：通过hedonic回归，控制了产品特征（features）的影响
- **纯价格变化**：反映了在相同质量水平下的价格变化

### 3.3 两者的区别

| 维度 | Traditional | Hedonic |
|------|------------|---------|
| **数据来源** | 实际价格 | 预测价格（hedonic回归） |
| **质量调整** | ❌ 无 | ✅ 有（通过特征控制） |
| **反映内容** | 实际市场价格变化 | 质量调整后的价格变化 |

---

## 四、与其他模型的对比

### 4.1 与Delta Model的对比

**Delta Model**：
- 直接建模log价格差值：`log(price_t+1) - log(price_t)`
- Traditional：实际log差值的均值
- Hedonic：预测log差值的均值

**Time Dummy Model**：
- 建模log价格，通过time dummy控制时间效应
- Traditional：实际价格的log差值均值
- Hedonic：预测价格的log差值均值

**关键区别**：
- Delta Model：直接建模价格变化
- Time Dummy Model：建模价格，通过time dummy捕获时间效应

### 4.2 与Basic Hedonic Model的对比

**Basic Hedonic Model**：
- 每个period独立训练模型
- 预测每个period的价格
- 计算相邻period的预测价格差值

**Time Dummy Model**：
- 相邻两个period pool在一起，训练一个模型
- 通过time dummy区分不同period
- 预测两个period的价格，计算差值

**关键区别**：
- Basic Hedonic：每个period独立模型
- Time Dummy：相邻period共享模型，通过time dummy区分

---

## 五、Time Dummy系数的含义

Time Dummy模型的time dummy系数（`lasso.coef_[-1]`）表示：
- **在控制所有产品特征后，从period 1到period 2的平均log价格变化**
- 这实际上就是**Hedonic Jevons Index**的理论值

如果time dummy系数为-0.05，意味着：
- 在相同产品特征下，period 2的log价格比period 1低0.05
- 即价格下降了约5%

---

## 六、实际例子

假设有一个季度对（2020 Q1 → 2020 Q2）：

**Traditional Jevons Index**：
- 计算所有产品在2020 Q1和2020 Q2的实际价格log差值
- 取均值：`mean(log(price_q2) - log(price_q1)) = -0.095`
- 表示实际价格平均下降了9.5%

**Hedonic Jevons Index**：
- 用time dummy模型预测所有产品在两个季度的价格
- 计算预测价格的log差值：`mean(log(pred_price_q2) - log(pred_price_q1)) = -0.095`
- 表示在质量调整后，价格平均下降了9.5%

**差异**：
- 如果Traditional和Hedonic不同，说明：
  - 产品特征（质量）的变化影响了价格
  - Hedonic通过控制特征，分离出了纯价格变化

---

## 七、总结

Time Dummy模型有Traditional和Hedonic两个Jevons Index的原因是：

1. **模型本质**：Time Dummy模型仍然是hedonic回归，可以预测质量调整后的价格
2. **数据可用性**：我们有实际价格数据，可以计算Traditional Index作为基准
3. **对比需要**：通过对比两个Index，可以评估质量调整的效果
4. **方法一致性**：与其他模型（Delta Model, Basic Hedonic）保持一致的计算方式

**关键点**：
- Traditional = 实际价格变化的均值（无质量调整）
- Hedonic = 预测价格变化的均值（有质量调整）
- 两者的差异反映了质量变化对价格的影响

---

*参考代码*：
- `Quarter/lasso_time_dummy_model.py` - 季度Time Dummy模型
- `annual/lasso_time_dummy_model_annual.py` - 年度Time Dummy模型


