"""
供应商供货量预测模型 - Version 2.1
基于统计特征的机器学习预测模型
新增功能：模型保存/加载、多线程预测、GPU加速支持
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SupplierPredictionModel:
    """供应商供货量预测模型"""
    
    def __init__(self, model_save_path='models/'):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.is_trained = False
        self.supplier_stats = {}
        self.model_save_path = model_save_path
        self.model_file = os.path.join(model_save_path, 'supplier_prediction_model.pkl')
        
        # 创建模型保存目录
        os.makedirs(model_save_path, exist_ok=True)
        
        # 尝试检测GPU支持
        self.use_gpu = self._check_gpu_support()
        
    def _check_gpu_support(self):
        """检测GPU支持"""
        try:
            # 检测XGBoost GPU支持
            import xgboost as xgb
            if hasattr(xgb, 'cuda') and xgb.cuda.is_cuda_available():
                print("✓ 检测到XGBoost GPU支持")
                return True
        except ImportError:
            pass
        
        print("⚠ 未检测到GPU支持，使用CPU模式")
        return False
        
    def save_model(self):
        """保存训练好的模型"""
        if not self.is_trained:
            print("⚠ 模型尚未训练，无法保存")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'supplier_stats': self.supplier_stats,
                'supply_history': self.supply_history,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, self.model_file)
            print(f"✓ 模型已保存到: {self.model_file}")
            return True
            
        except Exception as e:
            print(f"✗ 模型保存失败: {e}")
            return False
    
    def load_model(self):
        """加载预训练模型"""
        if not os.path.exists(self.model_file):
            print(f"⚠ 模型文件不存在: {self.model_file}")
            return False
        
        try:
            model_data = joblib.load(self.model_file)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.supplier_stats = model_data['supplier_stats']
            self.supply_history = model_data['supply_history']
            self.is_trained = model_data['is_trained']
            
            print(f"✓ 模型已从文件加载: {self.model_file}")
            return True
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return False
    
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
        
        # 使用tqdm显示进度
        for _, supplier_row in tqdm(supply_df.iterrows(), total=len(supply_df), desc="处理供应商"):
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
            n_jobs=-1  # 使用所有CPU核心
        )
        rf_model.fit(X_train, y_train)
        
        print("训练梯度提升模型...")
        if self.use_gpu:
            try:
                # 尝试使用GPU加速的XGBoost
                import xgboost as xgb
                gb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='hist',
                    device='cuda',  # 使用GPU
                    gpu_id=0
                )
                print("✓ 使用GPU加速XGBoost")
            except:
                # 回退到普通梯度提升
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                )
                print("回退到CPU梯度提升")
        else:
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
        
        # 自动保存模型
        self.save_model()
        
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
    
    def _predict_single_supplier_worker(self, args):
        """单个供应商预测的工作函数（用于并行处理）"""
        supplier_id, prediction_weeks = args
        try:
            return supplier_id, self.predict_supplier_supply(supplier_id, prediction_weeks)
        except Exception as e:
            # 出错时返回基于历史均值的预测
            if supplier_id in self.supplier_stats:
                avg_supply = self.supplier_stats[supplier_id]['平均值']
                std_supply = self.supplier_stats[supplier_id]['标准差']
                predictions = np.random.normal(avg_supply, std_supply, prediction_weeks)
                predictions = np.maximum(predictions, 0)
                return supplier_id, predictions
            else:
                return supplier_id, np.zeros(prediction_weeks)
    
    def batch_predict(self, supplier_ids, prediction_weeks, use_multithread=True, max_workers=None):
        """
        批量预测多个供应商的供货量
        
        参数:
        - supplier_ids: 供应商ID列表
        - prediction_weeks: 预测周数
        - use_multithread: 是否使用多线程
        - max_workers: 最大工作线程数
        
        返回:
        - 字典，键为supplier_id，值为预测数组
        """
        results = {}
        
        if use_multithread and len(supplier_ids) > 5:
            # 使用多线程并行预测
            if max_workers is None:
                max_workers = min(32, (os.cpu_count() or 1) + 4)
            
            # 准备任务列表
            tasks = [(supplier_id, prediction_weeks) for supplier_id in supplier_ids]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 使用tqdm显示进度
                futures = [executor.submit(self._predict_single_supplier_worker, task) for task in tasks]
                
                for future in tqdm(futures, desc="批量预测", leave=False):
                    supplier_id, predictions = future.result()
                    results[supplier_id] = predictions
        else:
            # 单线程顺序预测
            for supplier_id in tqdm(supplier_ids, desc="批量预测", leave=False):
                try:
                    predictions = self.predict_supplier_supply(supplier_id, prediction_weeks)
                    results[supplier_id] = predictions
                except Exception as e:
                    # 使用历史平均值作为备选
                    if supplier_id in self.supplier_stats:
                        avg_supply = self.supplier_stats[supplier_id]['平均值']
                        std_supply = self.supplier_stats[supplier_id]['标准差']
                        predictions = np.random.normal(avg_supply, std_supply, prediction_weeks)
                        predictions = np.maximum(predictions, 0)
                        results[supplier_id] = predictions
                    else:
                        results[supplier_id] = np.zeros(prediction_weeks)
        
        return results


# 全局模型实例
_global_model = None

def get_trained_model(force_retrain=False):
    """获取训练好的全局模型实例"""
    global _global_model
    
    if _global_model is None:
        _global_model = SupplierPredictionModel()
    
    # 首先尝试加载已有模型
    if not force_retrain and not _global_model.is_trained:
        loaded = _global_model.load_model()
        if not loaded:
            print("未找到预训练模型，开始训练新模型...")
            _global_model.train_model()
    elif force_retrain:
        print("强制重新训练模型...")
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

def predict_multiple_suppliers(supplier_ids, prediction_weeks, use_multithread=True):
    """
    批量预测多个供应商的供货量（对外接口）
    
    参数:
    - supplier_ids: 供应商ID列表  
    - prediction_weeks: 预测周数
    - use_multithread: 是否使用多线程
    
    返回:
    - 字典，键为supplier_id，值为预测数组
    """
    model = get_trained_model()
    return model.batch_predict(supplier_ids, prediction_weeks, use_multithread=use_multithread)


if __name__ == "__main__":
    # 测试代码
    print("开始测试ML预测模型...")
    
    # 获取训练好的模型（自动加载或训练）
    model = get_trained_model()
    
    # 测试预测
    test_suppliers = ['S001', 'S002', 'S003']
    test_weeks = 24
    
    print(f"\n测试预测 {test_suppliers}，{test_weeks} 周...")
    
    # 测试单个预测
    for supplier_id in test_suppliers:
        try:
            predictions = model.predict_supplier_supply(supplier_id, test_weeks)
            print(f"供应商 {supplier_id}:")
            print(f"  预测平均值: {np.mean(predictions):.2f}")
            print(f"  预测标准差: {np.std(predictions):.2f}")
            print(f"  预测范围: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
        except Exception as e:
            print(f"供应商 {supplier_id} 预测失败: {e}")
    
    # 测试批量预测（多线程）
    print(f"\n测试批量预测（多线程）...")
    batch_results = model.batch_predict(test_suppliers, test_weeks, use_multithread=True)
    
    for supplier_id, predictions in batch_results.items():
        print(f"供应商 {supplier_id}: 均值 {np.mean(predictions):.2f}, 标准差 {np.std(predictions):.2f}")
    
    print("\n✓ 测试完成!")
