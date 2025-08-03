"""
简化的供应商预测接口
专门用于替换Problem2.ipynb中的蒙特卡洛模拟
"""

import os
import sys
import numpy as np
import pandas as pd

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from supplier_prediction_model_v2 import SupplierPredictorV2
except ImportError:
    print("警告：无法导入ML模型，将使用简化预测方法")
    SupplierPredictorV2 = None

class SimpleSupplierPredictor:
    """简化的供应商预测器 - 专用于Problem2.ipynb"""
    
    def __init__(self):
        self.predictor = None
        self.initialized = False
        self._init_predictor()
    
    def _init_predictor(self):
        """初始化预测器"""
        if SupplierPredictorV2 is None:
            return
            
        try:
            self.predictor = SupplierPredictorV2()
            model_path = 'models/supplier_predictor_v2.pkl'
            
            if os.path.exists(model_path):
                self.predictor.load_model(model_path)
                self.predictor.load_data()
                self.initialized = True
                print("✓ ML预测模型已就绪")
            else:
                print("正在训练ML模型...")
                if self.predictor.train():
                    self.predictor.save_model(model_path)
                    self.initialized = True
                    print("✓ ML模型训练完成")
                else:
                    print("✗ ML模型训练失败，将使用简化方法")
                    
        except Exception as e:
            print(f"ML模型初始化失败: {e}")
            print("将使用简化预测方法")
    
    def predict_supplier_supply(self, supplier_id, num_weeks=4):
        """
        预测供应商供货量 - 简化接口
        
        Args:
            supplier_id: 供应商ID (如 'S229')
            num_weeks: 预测周数 (默认4周)
            
        Returns:
            list: 预测的供货量列表，长度为num_weeks
        """
        if self.initialized and self.predictor:
            try:
                # 使用ML模型预测
                result = self.predictor.predict_supplier(supplier_id, num_weeks)
                if result and 'predictions' in result:
                    return [pred['predicted_supply'] for pred in result['predictions']]
            except Exception as e:
                print(f"ML预测失败 ({supplier_id}): {e}")
        
        # 备用简化预测方法
        return self._simple_predict(supplier_id, num_weeks)
    
    def _simple_predict(self, supplier_id, num_weeks):
        """简化预测方法（备用）"""
        # 基于供应商ID的简单预测规则
        if supplier_id == 'S229':
            base = 1500  # S229比较稳定，均值1500左右
            return [base * (0.9 + 0.2 * np.random.random()) for _ in range(num_weeks)]
        elif supplier_id == 'S348':
            base = 300   # S348不太稳定，平均300左右
            return [base * (0.5 + 1.0 * np.random.random()) for _ in range(num_weeks)]
        elif supplier_id == 'S016':
            base = 30    # S016供货很少，平均30左右
            return [base * (0.3 + 1.4 * np.random.random()) for _ in range(num_weeks)]
        else:
            # 其他供应商使用默认值
            base = 100
            return [base * (0.5 + 1.0 * np.random.random()) for _ in range(num_weeks)]


# 全局预测器实例
_global_predictor = None

def get_predictor():
    """获取全局预测器实例"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = SimpleSupplierPredictor()
    return _global_predictor

def predict_supplier(supplier_id, num_weeks=4):
    """
    快速预测函数 - 最简接口
    
    Args:
        supplier_id: 供应商ID
        num_weeks: 预测周数
        
    Returns:
        list: 预测供货量列表
        
    Example:
        >>> predictions = predict_supplier('S229', 4)
        >>> print(predictions)  # [1587.7, 1554.5, 1641.7, 1615.9]
    """
    predictor = get_predictor()
    return predictor.predict_supplier_supply(supplier_id, num_weeks)

def batch_predict_suppliers(supplier_ids, num_weeks=4):
    """
    批量预测多个供应商
    
    Args:
        supplier_ids: 供应商ID列表
        num_weeks: 预测周数
        
    Returns:
        dict: {supplier_id: [预测值列表]}
    """
    predictor = get_predictor()
    results = {}
    
    for supplier_id in supplier_ids:
        results[supplier_id] = predictor.predict_supplier_supply(supplier_id, num_weeks)
    
    return results

def monte_carlo_with_ml(supplier_selection, target_capacity, simulation_weeks=24):
    """
    基于ML的蒙特卡洛模拟 - 专用于Problem2.ipynb
    
    Args:
        supplier_selection: 供应商选择DataFrame，包含'供应商ID'列
        target_capacity: 目标总产能
        simulation_weeks: 模拟周数
        
    Returns:
        dict: 模拟结果
    """
    print(f"ML蒙特卡洛模拟: 目标={target_capacity:,.0f}, 周数={simulation_weeks}")
    
    predictor = get_predictor()
    
    # 获取所有供应商的预测
    supplier_predictions = {}
    for _, supplier in supplier_selection.iterrows():
        supplier_id = supplier['供应商ID']
        predictions = predictor.predict_supplier_supply(supplier_id, simulation_weeks)
        supplier_predictions[supplier_id] = predictions
    
    # 计算每周总产能
    weekly_totals = []
    for week in range(simulation_weeks):
        week_total = sum([supplier_predictions[sid][week] 
                         for sid in supplier_predictions])
        weekly_totals.append(week_total)
    
    # 计算成功率
    successful_weeks = sum([1 for total in weekly_totals if total >= target_capacity])
    success_rate = successful_weeks / simulation_weeks
    
    return {
        'weekly_capacities': weekly_totals,
        'success_rate': success_rate,
        'supplier_predictions': supplier_predictions,
        'avg_capacity': np.mean(weekly_totals),
        'min_capacity': min(weekly_totals),
        'max_capacity': max(weekly_totals)
    }


# 演示和测试
if __name__ == "__main__":
    print("=" * 50)
    print("简化供应商预测接口测试")
    print("=" * 50)
    
    # 测试单个预测
    print("\n1. 测试单个供应商预测:")
    for supplier_id in ['S229', 'S348', 'S016']:
        predictions = predict_supplier(supplier_id, 4)
        print(f"  {supplier_id}: {[f'{p:.1f}' for p in predictions]}")
    
    # 测试批量预测
    print("\n2. 测试批量预测:")
    batch_results = batch_predict_suppliers(['S229', 'S348', 'S016'], 3)
    for supplier_id, predictions in batch_results.items():
        print(f"  {supplier_id}: {[f'{p:.1f}' for p in predictions]}")
    
    # 测试蒙特卡洛模拟
    print("\n3. 测试ML蒙特卡洛模拟:")
    test_suppliers = pd.DataFrame({
        '供应商ID': ['S229', 'S348', 'S016']
    })
    
    mc_result = monte_carlo_with_ml(test_suppliers, target_capacity=5000, simulation_weeks=4)
    print(f"  成功率: {mc_result['success_rate']:.2%}")
    print(f"  平均产能: {mc_result['avg_capacity']:.0f}")
    print(f"  周产能范围: {mc_result['min_capacity']:.0f} - {mc_result['max_capacity']:.0f}")
