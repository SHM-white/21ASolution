"""
使用全历史数据模型的简化预测接口
适配Problem2.ipynb中的蒙特卡洛模拟
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 全局预测器实例
_predictor = None

def _get_predictor():
    """获取或初始化预测器"""
    global _predictor
    
    if _predictor is None:
        from supplier_predictor_full_history import SupplierPredictorFullHistory
        _predictor = SupplierPredictorFullHistory(window_size=48)
        
        # 尝试加载已训练的模型
        if not _predictor.load_model('models/supplier_predictor_full_history.pkl'):
            print("模型文件不存在，开始训练新模型...")
            if _predictor.train_model():
                _predictor.save_model('models/supplier_predictor_full_history.pkl')
            else:
                print("✗ 模型训练失败")
                return None
    
    return _predictor

def predict_supplier(supplier_id, num_weeks=4):
    """
    预测供应商未来几周的供货量
    
    参数:
    - supplier_id: 供应商ID
    - num_weeks: 预测周数，默认4周
    
    返回:
    - predictions: 预测值列表，如果预测失败返回None
    """
    predictor = _get_predictor()
    if predictor is None:
        return None
    
    try:
        predictions = predictor.predict_supplier(supplier_id, num_weeks)
        return predictions
    except Exception as e:
        print(f"预测失败 ({supplier_id}): {e}")
        return None

def monte_carlo_with_ml(supplier_selection, target_weekly_capacity, simulation_weeks=24, num_simulations=100):
    """
    使用ML模型进行蒙特卡洛模拟
    
    参数:
    - supplier_selection: DataFrame，包含选中的供应商ID
    - target_weekly_capacity: 目标周产能
    - simulation_weeks: 模拟周数
    - num_simulations: 模拟次数
    
    返回:
    - 模拟结果字典
    """
    predictor = _get_predictor()
    if predictor is None:
        return None
    
    print(f"开始ML蒙特卡洛模拟...")
    print(f"  选中供应商: {len(supplier_selection)} 家")
    print(f"  目标产能: {target_weekly_capacity:,.0f}")
    print(f"  模拟周数: {simulation_weeks}")
    print(f"  模拟次数: {num_simulations}")
    
    success_count = 0
    all_weekly_capacities = []
    all_min_capacities = []
    all_avg_capacities = []
    
    supplier_ids = supplier_selection['供应商ID'].tolist()
    
    # 预测每个供应商的基线能力
    supplier_predictions = {}
    for supplier_id in supplier_ids:
        try:
            # 预测较长时间以获得更稳定的基线
            base_predictions = predict_supplier(supplier_id, 8)
            if base_predictions:
                supplier_predictions[supplier_id] = {
                    'baseline': np.mean(base_predictions),
                    'volatility': np.std(base_predictions) / np.mean(base_predictions) if np.mean(base_predictions) > 0 else 0.1
                }
            else:
                print(f"  供应商 {supplier_id} 预测失败，使用默认值")
                supplier_predictions[supplier_id] = {'baseline': 100, 'volatility': 0.2}
        except Exception as e:
            print(f"  供应商 {supplier_id} 预测出错: {e}")
            supplier_predictions[supplier_id] = {'baseline': 100, 'volatility': 0.2}
    
    print(f"  成功获取 {len(supplier_predictions)} 个供应商的预测基线")
    
    # 进行蒙特卡洛模拟
    for sim in range(num_simulations):
        weekly_capacities = []
        
        for week in range(simulation_weeks):
            weekly_total = 0
            
            for supplier_id in supplier_ids:
                if supplier_id in supplier_predictions:
                    baseline = supplier_predictions[supplier_id]['baseline']
                    volatility = supplier_predictions[supplier_id]['volatility']
                    
                    # 加入随机波动
                    random_factor = np.random.normal(1, volatility)
                    # 限制在合理范围内
                    random_factor = np.clip(random_factor, 0.5, 1.5)
                    
                    weekly_supply = baseline * random_factor
                    weekly_total += weekly_supply
            
            weekly_capacities.append(weekly_total)
        
        # 记录本次模拟结果
        min_weekly = min(weekly_capacities)
        avg_weekly = np.mean(weekly_capacities)
        
        all_weekly_capacities.append(weekly_capacities)
        all_min_capacities.append(min_weekly)
        all_avg_capacities.append(avg_weekly)
        
        # 判断是否成功（所有周都满足需求）
        if min_weekly >= target_weekly_capacity:
            success_count += 1
        
        if (sim + 1) % 20 == 0:
            current_success_rate = success_count / (sim + 1)
            print(f"    进度: {sim + 1}/{num_simulations}, 当前成功率: {current_success_rate:.2%}")
    
    # 计算最终结果
    success_rate = success_count / num_simulations
    avg_min_capacity = np.mean(all_min_capacities)
    avg_avg_capacity = np.mean(all_avg_capacities)
    
    result = {
        'success_rate': success_rate,
        'min_capacity': avg_min_capacity,
        'avg_capacity': avg_avg_capacity,
        'weekly_capacities': np.mean(all_weekly_capacities, axis=0).tolist(),
        'num_suppliers': len(supplier_ids),
        'simulation_details': {
            'num_simulations': num_simulations,
            'simulation_weeks': simulation_weeks,
            'target_capacity': target_weekly_capacity
        }
    }
    
    print(f"  最终成功率: {success_rate:.2%}")
    print(f"  平均最低产能: {avg_min_capacity:,.0f}")
    print(f"  平均平均产能: {avg_avg_capacity:,.0f}")
    
    return result

def test_new_model():
    """测试新模型的功能"""
    print("=" * 60)
    print("测试新的全历史预测模型")
    print("=" * 60)
    
    # 测试单个供应商预测
    test_suppliers = ['S229', 'S348', 'S001', 'S002']
    
    print("1. 测试单个供应商预测:")
    for supplier_id in test_suppliers:
        predictions = predict_supplier(supplier_id, 4)
        if predictions:
            avg_pred = np.mean(predictions)
            print(f"   ✓ {supplier_id}: 4周预测 = {[f'{p:.1f}' for p in predictions]}, 平均 = {avg_pred:.1f}")
        else:
            print(f"   ✗ {supplier_id}: 预测失败")
    
    # 测试蒙特卡洛模拟
    print(f"\n2. 测试蒙特卡洛模拟:")
    test_selection = pd.DataFrame({
        '供应商ID': ['S229', 'S348', 'S001']
    })
    
    mc_result = monte_carlo_with_ml(test_selection, 1000, 12, 50)
    if mc_result:
        print(f"   ✓ 模拟成功:")
        print(f"     成功率: {mc_result['success_rate']:.2%}")
        print(f"     最低产能: {mc_result['min_capacity']:,.0f}")
        print(f"     平均产能: {mc_result['avg_capacity']:,.0f}")
    else:
        print(f"   ✗ 模拟失败")

if __name__ == "__main__":
    test_new_model()
