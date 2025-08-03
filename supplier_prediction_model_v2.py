"""
供应商供货量预测模型 V2.0
直接基于历史数据的简化预测模型

主要功能：
1. 直接从数据表读取供应商历史供货数据
2. 基于时间序列特征进行预测
3. 简化接口：只需供应商ID和预测周数
4. 针对不同供应商的供货特征进行个性化预测
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

warnings.filterwarnings('ignore')

class SupplierPredictorV2:
    """供应商供货量预测器 V2.0"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.supplier_data = None
        self.week_columns = []
        self.is_trained = False
        
    def load_data(self):
        """加载供应商历史供货数据"""
        print("正在加载供应商历史供货数据...")
        
        try:
            # 读取供应商周供货数据（实际供货量）
            supply_file = "C/附件1 近5年402家供应商的相关数据.xlsx"
            if os.path.exists(supply_file):
                self.supplier_data = pd.read_excel(supply_file)
                print(f"✓ 成功加载供应商数据: {len(self.supplier_data)} 家供应商")
            else:
                # 备用数据源
                supply_file = "DataFrames/原材料转换为产品制造能力.xlsx"
                self.supplier_data = pd.read_excel(supply_file)
                print(f"✓ 成功加载备用供应商数据: {len(self.supplier_data)} 家供应商")
            
            # 获取周数据列
            self.week_columns = [col for col in self.supplier_data.columns if col.startswith('W')]
            print(f"✓ 发现 {len(self.week_columns)} 周的历史数据")
            
            return True
            
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return False
    
    def get_supplier_history(self, supplier_id):
        """获取指定供应商的历史供货数据"""
        supplier_row = self.supplier_data[self.supplier_data['供应商ID'] == supplier_id]
        
        if supplier_row.empty:
            print(f"警告: 未找到供应商 {supplier_id}")
            return None, None
        
        supplier_row = supplier_row.iloc[0]
        material_type = supplier_row['材料分类']
        
        # 获取历史供货数据
        history = []
        for col in self.week_columns:
            value = supplier_row[col]
            if pd.notna(value):
                history.append(float(value))
            else:
                history.append(0.0)
        
        return np.array(history), material_type
    
    def create_training_data(self):
        """创建训练数据"""
        print("正在创建训练数据...")
        
        features = []
        targets = []
        
        for _, supplier_row in self.supplier_data.iterrows():
            supplier_id = supplier_row['供应商ID']
            material_type = supplier_row['材料分类']
            
            # 获取该供应商的历史数据
            history = []
            for col in self.week_columns:
                value = supplier_row[col]
                if pd.notna(value):
                    history.append(float(value))
                else:
                    history.append(0.0)
            
            history = np.array(history)
            
            # 创建滑动窗口数据
            window_size = 48  # 使用48周历史数据预测下一周
            
            for i in range(window_size, len(history) - 1):
                # 输入特征：前48周的数据
                input_window = history[i-window_size:i]
                # 目标：下一周的供货量
                target = history[i]
                
                # 只有当目标值和历史数据都有效时才添加
                if target >= 0 and np.sum(input_window >= 0) >= window_size * 0.7:  # 至少70%的数据有效
                    
                    # 基础特征
                    feature_vector = list(input_window)  # 直接使用原始历史数据
                    
                    # 统计特征
                    valid_history = input_window[input_window >= 0]
                    if len(valid_history) > 0:
                        feature_vector.extend([
                            np.mean(valid_history),      # 均值
                            np.std(valid_history),       # 标准差
                            np.max(valid_history),       # 最大值
                            np.min(valid_history),       # 最小值
                            np.median(valid_history),    # 中位数
                        ])
                    else:
                        feature_vector.extend([0, 0, 0, 0, 0])
                    
                    # 趋势特征
                    if len(valid_history) >= 2:
                        trend = valid_history[-1] - valid_history[0]  # 趋势
                        volatility = np.std(valid_history) / (np.mean(valid_history) + 1e-8)  # 波动率
                        feature_vector.extend([trend, volatility])
                    else:
                        feature_vector.extend([0, 0])
                    
                    # 时间特征
                    feature_vector.extend([
                        i % 52,           # 年内周数
                        (i // 4) % 12,   # 月份
                        i,                # 绝对周数
                    ])
                    
                    # 材料类型编码
                    material_encode = {'A': 1, 'B': 2, 'C': 3}.get(material_type, 0)
                    feature_vector.append(material_encode)
                    
                    features.append(feature_vector)
                    targets.append(target)
        
        X = np.array(features)
        y = np.array(targets)
        
        print(f"✓ 创建训练数据完成: {len(X)} 个样本, {X.shape[1]} 个特征")
        print(f"  目标值统计: 均值={np.mean(y):.2f}, 中位数={np.median(y):.2f}, 最大值={np.max(y):.2f}")
        
        return X, y
    
    def train(self):
        """训练模型"""
        print("=" * 60)
        print("开始训练供应商供货量预测模型 V2.0")
        print("=" * 60)
        
        # 加载数据
        if not self.load_data():
            return False
        
        # 创建训练数据
        X, y = self.create_training_data()
        
        if len(X) == 0:
            print("✗ 无有效训练数据")
            return False
        
        print("正在训练模型...")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练随机森林模型
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # 模型评估
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"模型训练完成:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.3f}")
        
        self.is_trained = True
        return True
    
    def predict_supplier(self, supplier_id, num_weeks=4):
        """
        预测指定供应商的未来供货量
        
        Args:
            supplier_id: 供应商ID (如 'S229')
            num_weeks: 预测周数
            
        Returns:
            dict: 预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 获取供应商历史数据
        history, material_type = self.get_supplier_history(supplier_id)
        
        if history is None:
            return None
        
        # print(f"\n供应商 {supplier_id} ({material_type}类材料):")
        # print(f"  历史数据统计: 均值={np.mean(history[history>=0]):.2f}, 中位数={np.median(history[history>=0]):.2f}")
        # print(f"  最大值={np.max(history):.2f}, 非零周数={np.sum(history>0)}")
        
        # 使用最近48周数据作为输入
        recent_history = history[-48:]
        predictions = []
        
        # 滚动预测
        current_window = list(recent_history)
        
        for week in range(num_weeks):
            # 构建特征 - 确保至少有48周的数据
            if len(current_window) < 48:
                # 如果历史数据不足48周，用0填充前面的部分
                padded_window = [0] * (48 - len(current_window)) + current_window
                input_window = np.array(padded_window[-48:])
            else:
                input_window = np.array(current_window[-48:])
            
            feature_vector = list(input_window)
            
            # 统计特征
            valid_data = input_window[input_window >= 0]
            if len(valid_data) > 0:
                feature_vector.extend([
                    np.mean(valid_data),
                    np.std(valid_data),
                    np.max(valid_data),
                    np.min(valid_data),
                    np.median(valid_data),
                ])
            else:
                feature_vector.extend([0, 0, 0, 0, 0])
            
            # 趋势特征
            if len(valid_data) >= 2:
                trend = valid_data[-1] - valid_data[0]
                volatility = np.std(valid_data) / (np.mean(valid_data) + 1e-8)
                feature_vector.extend([trend, volatility])
            else:
                feature_vector.extend([0, 0])
            
            # 时间特征 (假设从第240周开始预测)
            current_week = 240 + week
            feature_vector.extend([
                current_week % 52,
                (current_week // 4) % 12,
                current_week,
            ])
            
            # 材料类型
            material_encode = {'A': 1, 'B': 2, 'C': 3}.get(material_type, 0)
            feature_vector.append(material_encode)
            
            # 预测
            X_pred = self.scaler.transform([feature_vector])
            prediction = self.model.predict(X_pred)[0]
            
            # 确保预测值为非负
            prediction = max(0, prediction)
            
            predictions.append({
                'week': week + 1,
                'predicted_supply': prediction
            })
            
            # 更新滑动窗口
            current_window.append(prediction)
        
        return {
            'supplier_id': supplier_id,
            'material_type': material_type,
            'historical_stats': {
                'mean': float(np.mean(history[history>=0])),
                'median': float(np.median(history[history>=0])),
                'max': float(np.max(history)),
                'active_weeks': int(np.sum(history > 0))
            },
            'predictions': predictions
        }
    
    def save_model(self, filepath='models/supplier_predictor_v2.pkl'):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'week_columns': self.week_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath='models/supplier_predictor_v2.pkl'):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.week_columns = model_data['week_columns']
        self.is_trained = True
        
        print(f"模型已加载: {filepath}")


def test_representative_suppliers():
    """测试代表性供应商的预测效果"""
    
    # 创建并训练模型
    predictor = SupplierPredictorV2()
    
    if not predictor.train():
        print("模型训练失败")
        return
    
    # 保存模型
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("测试代表性供应商预测效果")
    print("=" * 60)
    
    # 测试三个代表性供应商
    test_suppliers = ['S229', 'S348', 'S016']
    
    for supplier_id in test_suppliers:
        result = predictor.predict_supplier(supplier_id, num_weeks=4)
        
        if result:
            print(f"\n【{supplier_id}】预测结果:")
            print(f"  材料类型: {result['material_type']}")
            print(f"  历史平均: {result['historical_stats']['mean']:.2f}")
            print(f"  历史中位数: {result['historical_stats']['median']:.2f}")
            print(f"  历史最大值: {result['historical_stats']['max']:.2f}")
            print(f"  活跃周数: {result['historical_stats']['active_weeks']}")
            print("  未来4周预测:")
            
            for pred in result['predictions']:
                print(f"    第{pred['week']}周: {pred['predicted_supply']:.2f}")
        else:
            print(f"\n【{supplier_id}】: 无法获取数据")


# 快速调用函数
def quick_predict(supplier_id, num_weeks=4, model_path='models/supplier_predictor_v2.pkl'):
    """
    快速预测函数 - 简化调用接口
    
    Args:
        supplier_id: 供应商ID
        num_weeks: 预测周数
        model_path: 模型路径
        
    Returns:
        list: 预测结果列表
    """
    try:
        predictor = SupplierPredictorV2()
        
        # 尝试加载已有模型
        if os.path.exists(model_path):
            predictor.load_model(model_path)
            predictor.load_data()  # 还需要加载数据
        else:
            # 重新训练
            if not predictor.train():
                return None
            predictor.save_model(model_path)
        
        # 进行预测
        result = predictor.predict_supplier(supplier_id, num_weeks)
        
        if result:
            return [pred['predicted_supply'] for pred in result['predictions']]
        else:
            return None
            
    except Exception as e:
        print(f"预测失败: {e}")
        return None


if __name__ == "__main__":
    # 测试代表性供应商
    test_representative_suppliers()
