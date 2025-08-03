# 供应商供货量预测模型使用说明

## 概述

本项目实现了一个基于机器学习的供应商供货量预测模型，可以替换传统的蒙特卡洛随机模拟方法，提供更精准的供货量预测和风险评估。

## 文件结构

```
├── supplier_prediction_model.py    # 核心预测模型实现
├── Problem2.ipynb                  # 主要分析notebook，包含集成代码
├── DataFrames/                     # 训练数据文件夹
│   ├── 原材料转换为产品制造能力.xlsx
│   ├── 供应商可靠性综合评估.xlsx
│   ├── 供应商基本特征分析.xlsx
│   └── ...
└── models/                         # 模型保存文件夹
    └── supplier_predictor.pkl      # 训练好的模型文件
```

## 核心功能

### 1. SupplierSupplyPredictor 类

主要的预测器类，提供以下功能：

- **模型训练**: 基于历史数据训练XGBoost+时序特征的混合模型
- **供货量预测**: 预测指定供应商在未来几周的供货量
- **不确定性量化**: 提供置信区间和风险评估
- **模型持久化**: 支持模型保存和加载

### 2. 关键函数

#### `create_and_train_predictor()`
创建并训练预测模型
```python
predictor = create_and_train_predictor(
    data_folder='DataFrames',
    model_save_path='models/supplier_predictor.pkl'
)
```

#### `predict_supplier_capacity_enhanced()`
增强的供货能力预测函数，可直接替换蒙特卡洛模拟
```python
results = predict_supplier_capacity_enhanced(
    supplier_data=selected_suppliers,
    target_capacity=28200,
    simulation_weeks=24,
    num_simulations=1000,
    predictor=trained_predictor
)
```

#### `replace_monte_carlo_with_ml_prediction()`
直接替换原蒙特卡洛模拟的函数，保持相同接口
```python
# 原来的调用
mc_results, min_suppliers = monte_carlo_minimum_suppliers(...)

# 新的调用（接口完全相同）
ml_results, min_suppliers = replace_monte_carlo_with_ml_prediction(...)
```

## 使用步骤

### 步骤1：数据准备
确保DataFrames文件夹中包含以下数据文件：
- 原材料转换为产品制造能力.xlsx
- 供应商可靠性综合评估.xlsx  
- 供应商基本特征分析.xlsx
- 供应商统计数据离散系数.xlsx

### 步骤2：训练模型
```python
from supplier_prediction_model import create_and_train_predictor

# 训练模型
predictor = create_and_train_predictor()
```

### 步骤3：使用预测
```python
# 方法1：直接替换蒙特卡洛模拟
from supplier_prediction_model import replace_monte_carlo_with_ml_prediction

results, min_suppliers = replace_monte_carlo_with_ml_prediction(
    supplier_data, target_capacity=28200, success_rate=0.95
)

# 方法2：单独预测某个供应商
predictions = predictor.predict_supplier_supply(
    supplier_id='S001',
    material_type='A', 
    historical_supplies=[100, 120, 110, 105],
    num_weeks=4
)
```

## 模型特点

### 1. 多层次特征工程
- **时序特征**: 4周滑动窗口历史数据
- **统计特征**: 均值、标准差、趋势、波动率
- **时间特征**: 周、月、季度、年度周期
- **供应商特征**: 可靠性评分、市场占有率、历史表现

### 2. 不确定性量化
- 提供预测的置信区间
- 量化供货风险因子
- 评估供货不足概率

### 3. 个性化建模
- 每个供应商有独立的预测参数
- 考虑材料类型差异
- 动态调整可靠性权重

## 与传统方法对比

| 特性 | 传统蒙特卡洛 | ML增强预测 |
|------|-------------|------------|
| 预测依据 | 随机采样 | 历史模式学习 |
| 精确性 | 中等 | 高 |
| 个性化 | 无 | 强 |
| 训练时间 | 无 | 中等 |
| 预测速度 | 慢 | 快 |
| 可解释性 | 低 | 中 |

## 性能指标

模型在测试集上的表现：
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差  
- **R²**: 决定系数
- **特征重要性**: 显示关键预测因子

## 风险评估

模型提供多维度风险评估：
- **波动性风险**: 基于历史波动率
- **供货不足概率**: 统计模型计算
- **可靠性风险**: 基于供应商历史表现
- **整体风险等级**: 综合评估结果

## 使用建议

### 1. 数据质量
- 确保历史数据完整性
- 定期更新训练数据
- 验证数据一致性

### 2. 模型更新
- 建议每季度重新训练模型
- 监控预测准确性
- 根据实际情况调整参数

### 3. 风险管理
- 结合置信区间制定采购计划
- 关注高风险供应商
- 建立应急预案

## 扩展功能

### 1. 集成到现有系统
模型提供标准接口，可以轻松集成到现有的供应链管理系统中。

### 2. 实时预测
支持在线预测，可以根据最新数据实时更新预测结果。

### 3. 多目标优化
除了供货量预测，还可以扩展到成本优化、风险最小化等多目标场景。

## 技术支持

如有问题，请检查：
1. 数据文件是否完整
2. Python环境依赖是否安装
3. 模型文件是否正确生成

## 更新日志

- v1.0: 基础预测模型实现
- v1.1: 增加不确定性量化
- v1.2: 优化特征工程和风险评估
