"""
完整48周历史的供应商预测模型
基于所有有效供货情况，使用全历史数据进行训练和预测
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class SupplierPredictorFullHistory:
    """基于全历史数据的供应商预测模型"""
    
    def __init__(self, window_size=48):
        """
        初始化预测模型
        
        参数:
        - window_size: 历史窗口大小（周数），默认48周
        """
        self.window_size = window_size
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.material_encoders = {'A': 0, 'B': 1, 'C': 2}
        self.is_trained = False
        
    def load_supplier_data(self):
        """加载供应商原始数据"""
        try:
            # 加载原始供货数据
            supplier_supply = pd.read_excel('C/附件1 近5年402家供应商的相关数据.xlsx', 
                                           sheet_name='供应商的供货量（m³）')
            
            print(f"✓ 加载供应商数据: {supplier_supply.shape}")
            print(f"  供应商数量: {len(supplier_supply)}")
            print(f"  材料类型: {supplier_supply['材料分类'].unique()}")
            
            return supplier_supply
            
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return None
    
    def prepare_training_data(self, supplier_data):
        """
        准备训练数据 - 使用所有有效供货情况
        """
        print(f"开始准备训练数据（窗口大小: {self.window_size}周）...")
        
        # 获取周数据列
        week_columns = [col for col in supplier_data.columns if col.startswith('W')]
        total_weeks = len(week_columns)
        print(f"  总周数: {total_weeks}")
        
        X_data = []
        y_data = []
        supplier_ids = []
        
        for idx, row in supplier_data.iterrows():
            supplier_id = row['供应商ID']
            material_type = row['材料分类']
            
            # 获取该供应商的周供货数据
            weekly_data = row[week_columns].values
            
            # 过滤有效数据（非零、非缺失）
            valid_weeks = []
            for i, value in enumerate(weekly_data):
                if pd.notna(value) and value > 0:
                    valid_weeks.append((i, value))
            
            # 如果有效数据不足，跳过
            if len(valid_weeks) < self.window_size + 1:
                continue
            
            # 使用滑动窗口创建训练样本
            for i in range(len(valid_weeks) - self.window_size):
                # 获取历史窗口数据
                window_data = [valid_weeks[j][1] for j in range(i, i + self.window_size)]
                
                # 目标值（下一周的供货量）
                target_value = valid_weeks[i + self.window_size][1]
                
                # 计算统计特征
                window_array = np.array(window_data)
                mean_val = np.mean(window_array)
                std_val = np.std(window_array)
                max_val = np.max(window_array)
                min_val = np.min(window_array)
                median_val = np.median(window_array)
                
                # 趋势特征
                trend = np.polyfit(range(len(window_array)), window_array, 1)[0]
                volatility = std_val / mean_val if mean_val > 0 else 0
                
                # 时间特征
                current_week_idx = valid_weeks[i + self.window_size - 1][0]
                week_in_year = (current_week_idx % 48) + 1  # 假设每年48周
                month = ((current_week_idx % 48) // 4) + 1  # 每4周一个月
                
                # 材料类型编码
                material_encoded = self.material_encoders.get(material_type, 0)
                
                # 构建特征向量
                features = list(window_data) + [
                    mean_val, std_val, max_val, min_val, median_val,  # 统计特征
                    trend, volatility,  # 趋势特征
                    week_in_year, month, current_week_idx,  # 时间特征
                    material_encoded  # 材料特征
                ]
                
                X_data.append(features)
                y_data.append(target_value)
                supplier_ids.append(supplier_id)
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"✓ 训练数据准备完成:")
        print(f"  样本数量: {len(X)}")
        print(f"  特征数量: {X.shape[1] if len(X) > 0 else 0}")
        print(f"  涉及供应商: {len(set(supplier_ids))}")
        
        return X, y, supplier_ids
    
    def train_model(self, supplier_data=None):
        """训练预测模型"""
        if supplier_data is None:
            supplier_data = self.load_supplier_data()
        
        if supplier_data is None:
            print("✗ 无法加载数据，训练失败")
            return False
        
        print("=" * 60)
        print("开始训练完整历史预测模型")
        print("=" * 60)
        
        # 准备训练数据
        X, y, supplier_ids = self.prepare_training_data(supplier_data)
        
        if len(X) == 0:
            print("✗ 没有有效的训练数据")
            return False
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        print("开始训练随机森林模型...")
        self.model.fit(X_train_scaled, y_train)
        
        # 模型评估
        y_pred = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n模型性能评估:")
        print(f"  平均绝对误差 (MAE): {mae:.2f}")
        print(f"  均方根误差 (RMSE): {rmse:.2f}")
        print(f"  决定系数 (R²): {r2:.4f}")
        
        # 特征重要性分析
        feature_names = [f'W{i+1}' for i in range(self.window_size)] + [
            'mean', 'std', 'max', 'min', 'median',
            'trend', 'volatility', 'week_in_year', 'month', 'week_idx', 'material'
        ]
        
        importances = self.model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n前10个重要特征:")
        for i, (name, importance) in enumerate(top_features):
            print(f"  {i+1:2d}. {name}: {importance:.4f}")
        
        self.is_trained = True
        return True
    
    def predict_supplier(self, supplier_id, num_weeks=4, supplier_data=None):
        """
        预测指定供应商未来几周的供货量
        """
        if not self.is_trained:
            print("✗ 模型未训练")
            return None
        
        if supplier_data is None:
            supplier_data = self.load_supplier_data()
        
        if supplier_data is None:
            return None
        
        # 查找供应商数据
        supplier_row = supplier_data[supplier_data['供应商ID'] == supplier_id]
        if supplier_row.empty:
            print(f"✗ 找不到供应商: {supplier_id}")
            return None
        
        supplier_row = supplier_row.iloc[0]
        material_type = supplier_row['材料分类']
        
        # 获取历史数据
        week_columns = [col for col in supplier_data.columns if col.startswith('W')]
        weekly_data = supplier_row[week_columns].values
        
        # 获取最近的有效数据
        valid_data = []
        for value in reversed(weekly_data):
            if pd.notna(value) and value > 0:
                valid_data.insert(0, value)
                if len(valid_data) >= self.window_size:
                    break
        
        if len(valid_data) < self.window_size:
            print(f"✗ 供应商 {supplier_id} 历史数据不足")
            return None
        
        # 取最近的window_size周数据
        recent_data = valid_data[-self.window_size:]
        predictions = []
        
        # 递归预测
        for week in range(num_weeks):
            # 计算特征
            window_array = np.array(recent_data)
            mean_val = np.mean(window_array)
            std_val = np.std(window_array)
            max_val = np.max(window_array)
            min_val = np.min(window_array)
            median_val = np.median(window_array)
            
            trend = np.polyfit(range(len(window_array)), window_array, 1)[0]
            volatility = std_val / mean_val if mean_val > 0 else 0
            
            # 时间特征（模拟）
            week_in_year = ((len(weekly_data) + week) % 48) + 1
            month = ((len(weekly_data) + week) % 48 // 4) + 1
            week_idx = len(weekly_data) + week
            
            material_encoded = self.material_encoders.get(material_type, 0)
            
            features = list(recent_data) + [
                mean_val, std_val, max_val, min_val, median_val,
                trend, volatility, week_in_year, month, week_idx, material_encoded
            ]
            
            # 预测
            features_scaled = self.scaler.transform([features])
            pred_value = self.model.predict(features_scaled)[0]
            
            # 确保预测值为正
            pred_value = max(pred_value, 0)
            
            predictions.append(pred_value)
            
            # 更新历史数据（滑动窗口）
            recent_data = recent_data[1:] + [pred_value]
        
        return predictions
    
    def save_model(self, filepath='models/supplier_predictor_full_history.pkl'):
        """保存模型"""
        if not self.is_trained:
            print("✗ 模型未训练，无法保存")
            return False
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'window_size': self.window_size,
            'material_encoders': self.material_encoders,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ 模型已保存到: {filepath}")
        return True
    
    def load_model(self, filepath='models/supplier_predictor_full_history.pkl'):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.window_size = model_data['window_size']
            self.material_encoders = model_data['material_encoders']
            self.is_trained = model_data['is_trained']
            
            print(f"✓ 模型已从 {filepath} 加载")
            return True
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return False

def main():
    """主函数 - 训练和测试模型"""
    # 创建预测器
    predictor = SupplierPredictorFullHistory(window_size=48)
    
    # 训练模型
    if predictor.train_model():
        # 保存模型
        predictor.save_model()
        
        # 测试预测
        test_suppliers = ['S229', 'S348', 'S016']
        print(f"\n" + "=" * 60)
        print("测试预测功能")
        print("=" * 60)
        
        for supplier_id in test_suppliers:
            predictions = predictor.predict_supplier(supplier_id, 4)
            if predictions:
                print(f"✓ {supplier_id}: {[f'{p:.1f}' for p in predictions]}")
            else:
                print(f"✗ {supplier_id}: 预测失败")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
