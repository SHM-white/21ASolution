"""
供应商供货量预测模型 - Version 2.0
基于统计特征的机器学习预测模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SupplierPredictionModel:
    """供应商供货量预测模型"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.is_trained = False
        self.supplier_stats = {}
        
    def load_data(self):
        """加载训练数据"""
        print("加载数据文件...")
        
        # 1. 加载供应商统计特征数据
        stat_df = pd.read_excel('DataFrames/供应商统计数据离散系数.xlsx', header=None)
        # 第3行(索引2)是列名，第4行(索引3)开始是数据
        headers = stat_df.iloc[2].tolist()
        stats_data = stat_df.iloc[3:].copy()
        stats_data.columns = headers
        stats_data = stats_data.reset_index(drop=True)
        
        # 清理数据类型
        numeric_columns = ['计数', '最小值', '最大值', '平均值', '平均值误差', '标准差', '方差', 
                          '偏度', '峰度', '25%分位数', '50%分位数', '75%分位数', '85%分位数', '90%分位数', '变异系数']
        
        for col in numeric_columns:
            stats_data[col] = pd.to_numeric(stats_data[col], errors='coerce')
        
        print(f"统计特征数据: {stats_data.shape}")
        
        # 2. 加载原始供货数据
        supply_df = pd.read_excel('C/附件1 近5年402家供应商的相关数据.xlsx', 
                                 sheet_name='供应商的供货量（m³）')
        print(f"原始供货数据: {supply_df.shape}")
        
        return stats_data, supply_df
    
    def prepare_training_data(self, stats_data, supply_df):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 获取周数据列
        week_columns = [col for col in supply_df.columns if col.startswith('W')]
        print(f"发现 {len(week_columns)} 个周数据列")
        
        training_samples = []
        
        for _, supplier_row in supply_df.iterrows():
            supplier_id = supplier_row['供应商ID']
            material_type = supplier_row['材料分类']
            
            # 获取该供应商的统计特征
            supplier_stats = stats_data[stats_data['供应商统计数据'] == supplier_id]
            if supplier_stats.empty:
                continue
                
            stats_row = supplier_stats.iloc[0]
            
            # 提取统计特征作为模型输入
            features = {
                'material_type_A': 1 if material_type == 'A' else 0,
                'material_type_B': 1 if material_type == 'B' else 0,
                'material_type_C': 1 if material_type == 'C' else 0,
                'count': stats_row['计数'],
                'mean': stats_row['平均值'],
                'std': stats_row['标准差'],
                'variance': stats_row['方差'],
                'skewness': stats_row['偏度'] if pd.notna(stats_row['偏度']) else 0,
                'kurtosis': stats_row['峰度'] if pd.notna(stats_row['峰度']) else 0,
                'min_val': stats_row['最小值'],
                'max_val': stats_row['最大值'],
                'q25': stats_row['25%分位数'],
                'q50': stats_row['50%分位数'],
                'q75': stats_row['75%分位数'],
                'q85': stats_row['85%分位数'],
                'q90': stats_row['90%分位数'],
                'cv': stats_row['变异系数']
            }
            
            # 为每个周创建训练样本（使用滑动窗口）
            week_values = supplier_row[week_columns].values
            
            # 使用滑动窗口：前N周预测后1周
            window_size = 8  # 使用8周历史数据预测下一周
            
            for i in range(window_size, len(week_values)):
                # 历史窗口特征
                window_data = week_values[i-window_size:i]
                
                sample_features = features.copy()
                sample_features.update({
                    f'lag_{j+1}': window_data[-(j+1)] for j in range(min(4, window_size))  # 最近4周
                })
                sample_features.update({
                    'window_mean': np.mean(window_data),
                    'window_std': np.std(window_data),
                    'window_trend': (window_data[-1] - window_data[0]) / window_size,
                    'week_in_year': i % 52  # 季节性特征
                })
                
                # 目标值
                target = week_values[i]
                
                training_samples.append({
                    'supplier_id': supplier_id,
                    'week_index': i,
                    'features': sample_features,
                    'target': target
                })
        
        print(f"生成 {len(training_samples)} 个训练样本")
        
        # 转换为DataFrame
        features_list = []
        targets = []
        
        for sample in training_samples:
            features_list.append(sample['features'])
            targets.append(sample['target'])
        
        features_df = pd.DataFrame(features_list)
        targets = np.array(targets)
        
        # 填充缺失值
        features_df = features_df.fillna(0)
        
        print(f"特征矩阵形状: {features_df.shape}")
        print(f"目标向量长度: {len(targets)}")
        
        return features_df, targets, training_samples
    
    def train_model(self):
        """训练预测模型"""
        print("开始训练模型...")
        
        # 加载数据
        stats_data, supply_df = self.load_data()
        
        # 准备训练数据
        X, y, training_samples = self.prepare_training_data(stats_data, supply_df)
        
        # 保存特征列名
        self.features = X.columns.tolist()
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 训练集成模型
        print("训练随机森林模型...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        print("训练梯度提升模型...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # 集成预测
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
        
        # 评估模型
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"模型性能:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: {np.sqrt(mse):.2f}")
        
        # 保存训练好的模型
        self.model = {
            'rf': rf_model,
            'gb': gb_model,
            'weights': [0.6, 0.4]
        }
        
        # 保存供应商统计信息用于预测
        for _, row in stats_data.iterrows():
            supplier_id = row['供应商统计数据']
            self.supplier_stats[supplier_id] = row.to_dict()
        
        # 保存供货历史数据用于特征工程
        self.supply_history = {}
        week_columns = [col for col in supply_df.columns if col.startswith('W')]
        for _, row in supply_df.iterrows():
            supplier_id = row['供应商ID']
            self.supply_history[supplier_id] = {
                'material_type': row['材料分类'],
                'history': row[week_columns].values
            }
        
        self.is_trained = True
        print("✓ 模型训练完成!")
        
        return {
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def predict_supplier_supply(self, supplier_id, prediction_weeks, start_week=240):
        """
        预测指定供应商在未来几周的供货量
        
        参数:
        - supplier_id: 供应商ID
        - prediction_weeks: 预测周数
        - start_week: 开始预测的周索引
        
        返回:
        - predictions: 预测结果数组（包含不确定性）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train_model() 方法")
        
        if supplier_id not in self.supplier_stats:
            raise ValueError(f"供应商 {supplier_id} 不在训练数据中")
        
        print(f"预测供应商 {supplier_id} 未来 {prediction_weeks} 周的供货量...")
        
        # 获取供应商的统计特征和历史数据
        stats = self.supplier_stats[supplier_id]
        history_info = self.supply_history[supplier_id]
        material_type = history_info['material_type']
        history = history_info['history']
        
        predictions = []
        current_history = history.copy()  # 复制历史数据
        
        for week in range(prediction_weeks):
            # 构建特征
            features = {
                'material_type_A': 1 if material_type == 'A' else 0,
                'material_type_B': 1 if material_type == 'B' else 0,
                'material_type_C': 1 if material_type == 'C' else 0,
                'count': stats['计数'],
                'mean': stats['平均值'],
                'std': stats['标准差'],
                'variance': stats['方差'],
                'skewness': stats['偏度'] if pd.notna(stats['偏度']) else 0,
                'kurtosis': stats['峰度'] if pd.notna(stats['峰度']) else 0,
                'min_val': stats['最小值'],
                'max_val': stats['最大值'],
                'q25': stats['25%分位数'],
                'q50': stats['50%分位数'],
                'q75': stats['75%分位数'],
                'q85': stats['85%分位数'],
                'q90': stats['90%分位数'],
                'cv': stats['变异系数']
            }
            
            # 最近的历史数据特征
            recent_data = current_history[-8:]  # 最近8周
            features.update({
                f'lag_{j+1}': recent_data[-(j+1)] for j in range(min(4, len(recent_data)))
            })
            features.update({
                'window_mean': np.mean(recent_data),
                'window_std': np.std(recent_data),
                'window_trend': (recent_data[-1] - recent_data[0]) / len(recent_data),
                'week_in_year': (start_week + week) % 52
            })
            
            # 确保特征顺序与训练时一致
            feature_vector = []
            for feat_name in self.features:
                feature_vector.append(features.get(feat_name, 0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # 集成预测
            rf_pred = self.model['rf'].predict(feature_vector_scaled)[0]
            gb_pred = self.model['gb'].predict(feature_vector_scaled)[0]
            base_prediction = (self.model['weights'][0] * rf_pred + 
                             self.model['weights'][1] * gb_pred)
            
            # 添加不确定性（基于历史波动性）
            uncertainty_factor = stats['变异系数'] if pd.notna(stats['变异系数']) else 0.2
            noise = np.random.normal(0, base_prediction * uncertainty_factor * 0.3)
            final_prediction = max(0, base_prediction + noise)  # 确保非负
            
            predictions.append(final_prediction)
            
            # 更新历史数据（用于下一周预测）
            current_history = np.append(current_history, final_prediction)
        
        return np.array(predictions)
    
    def batch_predict(self, supplier_ids, prediction_weeks):
        """
        批量预测多个供应商的供货量
        
        参数:
        - supplier_ids: 供应商ID列表
        - prediction_weeks: 预测周数
        
        返回:
        - 字典，键为supplier_id，值为预测数组
        """
        results = {}
        
        print(f"批量预测 {len(supplier_ids)} 个供应商，{prediction_weeks} 周...")
        
        for i, supplier_id in enumerate(supplier_ids):
            try:
                predictions = self.predict_supplier_supply(supplier_id, prediction_weeks)
                results[supplier_id] = predictions
                
                if (i + 1) % 20 == 0:  # 每20个输出一次进度
                    print(f"  已完成 {i + 1}/{len(supplier_ids)} 个供应商")
                    
            except Exception as e:
                print(f"  警告：供应商 {supplier_id} 预测失败: {e}")
                # 使用历史平均值作为备选
                if supplier_id in self.supplier_stats:
                    avg_supply = self.supplier_stats[supplier_id]['平均值']
                    std_supply = self.supplier_stats[supplier_id]['标准差']
                    # 生成基于历史均值和标准差的随机预测
                    predictions = np.random.normal(avg_supply, std_supply, prediction_weeks)
                    predictions = np.maximum(predictions, 0)  # 确保非负
                    results[supplier_id] = predictions
                else:
                    results[supplier_id] = np.zeros(prediction_weeks)
        
        print(f"✓ 批量预测完成")
        return results


# 全局模型实例
_global_model = None

def get_trained_model():
    """获取训练好的全局模型实例"""
    global _global_model
    if _global_model is None or not _global_model.is_trained:
        _global_model = SupplierPredictionModel()
        _global_model.train_model()
    return _global_model

def predict_single_supplier(supplier_id, prediction_weeks):
    """
    预测单个供应商的供货量（对外接口）
    
    参数:
    - supplier_id: 供应商ID
    - prediction_weeks: 预测周数
    
    返回:
    - 预测结果数组（包含不确定性）
    """
    model = get_trained_model()
    return model.predict_supplier_supply(supplier_id, prediction_weeks)

def predict_multiple_suppliers(supplier_ids, prediction_weeks):
    """
    批量预测多个供应商的供货量（对外接口）
    
    参数:
    - supplier_ids: 供应商ID列表  
    - prediction_weeks: 预测周数
    
    返回:
    - 字典，键为supplier_id，值为预测数组
    """
    model = get_trained_model()
    return model.batch_predict(supplier_ids, prediction_weeks)


if __name__ == "__main__":
    # 测试代码
    print("开始测试ML预测模型...")
    
    # 训练模型
    model = SupplierPredictionModel()
    training_result = model.train_model()
    
    print(f"\n训练结果: {training_result}")
    
    # 测试预测
    test_suppliers = ['S001', 'S002', 'S003']
    test_weeks = 24
    
    print(f"\n测试预测 {test_suppliers}，{test_weeks} 周...")
    
    for supplier_id in test_suppliers:
        try:
            predictions = model.predict_supplier_supply(supplier_id, test_weeks)
            print(f"供应商 {supplier_id}:")
            print(f"  预测平均值: {np.mean(predictions):.2f}")
            print(f"  预测标准差: {np.std(predictions):.2f}")
            print(f"  预测范围: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
        except Exception as e:
            print(f"供应商 {supplier_id} 预测失败: {e}")
    
    print("\n✓ 测试完成!")
