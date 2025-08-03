"""
蒙特卡洛模拟替换函数
使用机器学习模型替代随机模拟的供货量生成
"""

import numpy as np
import pandas as pd
from supplier_prediction_model_v2 import SupplierPredictorV2, quick_predict

def monte_carlo_minimum_suppliers_ml(supplier_selection, target_capacity=28800000, 
                                   simulation_weeks=24, num_simulations=1000):
    """
    基于机器学习的供应商最少数量确定（替换蒙特卡洛模拟）
    
    Args:
        supplier_selection: 包含供应商信息的DataFrame
        target_capacity: 目标总产能
        simulation_weeks: 模拟周数
        num_simulations: 模拟次数（保持接口一致，但实际使用ML预测）
        
    Returns:
        dict: 包含成功率、最少供应商数量、详细结果的字典
    """
    print(f"使用ML模型进行供应商最少数量分析...")
    print(f"目标产能: {target_capacity:,.0f}, 分析周数: {simulation_weeks}")
    
    # 加载或训练预测模型
    try:
        predictor = SupplierPredictorV2()
        model_path = 'models/supplier_predictor_v2.pkl'
        
        try:
            predictor.load_model(model_path)
            predictor.load_data()
            print("✓ 已加载预训练模型")
        except:
            print("正在训练新模型...")
            predictor.train()
            predictor.save_model(model_path)
            print("✓ 模型训练完成")
            
    except Exception as e:
        print(f"模型加载/训练失败: {e}")
        print("使用传统蒙特卡洛方法")
        return monte_carlo_fallback(supplier_selection, target_capacity, simulation_weeks)
    
    results = {
        'success_rates': [],
        'minimum_suppliers': 0,
        'detailed_results': [],
        'weekly_predictions': {}
    }
    
    # 按供应商数量递增测试
    suppliers = supplier_selection.copy()
    
    for num_suppliers in range(1, min(51, len(suppliers) + 1)):  # 最多测试50家
        current_suppliers = suppliers.head(num_suppliers)
        
        print(f"\n测试 {num_suppliers} 家供应商...")
        
        # 为每家供应商生成预测
        weekly_totals = []
        supplier_predictions = {}
        
        for _, supplier in current_suppliers.iterrows():
            supplier_id = supplier['供应商ID']
            
            # 使用ML模型预测
            predictions = predictor.predict_supplier(supplier_id, num_weeks=simulation_weeks)
            
            if predictions:
                weekly_supplies = [pred['predicted_supply'] for pred in predictions['predictions']]
                supplier_predictions[supplier_id] = weekly_supplies
            else:
                # 如果预测失败，使用历史平均值
                avg_supply = supplier.get('平均周制造能力', 0)
                weekly_supplies = [avg_supply * (0.8 + 0.4 * np.random.random()) 
                                 for _ in range(simulation_weeks)]
                supplier_predictions[supplier_id] = weekly_supplies
        
        # 计算每周总供应量
        for week in range(simulation_weeks):
            week_total = sum([supplier_predictions[sid][week] 
                            for sid in supplier_predictions])
            weekly_totals.append(week_total)
        
        # 计算成功率（满足目标产能的周数比例）
        successful_weeks = sum([1 for total in weekly_totals if total >= target_capacity])
        success_rate = successful_weeks / simulation_weeks
        
        results['success_rates'].append(success_rate)
        results['detailed_results'].append({
            'num_suppliers': num_suppliers,
            'success_rate': success_rate,
            'avg_weekly_capacity': np.mean(weekly_totals),
            'min_weekly_capacity': min(weekly_totals),
            'max_weekly_capacity': max(weekly_totals),
            'weekly_totals': weekly_totals
        })
        
        results['weekly_predictions'][num_suppliers] = supplier_predictions
        
        print(f"  成功率: {success_rate:.2%}")
        print(f"  平均周产能: {np.mean(weekly_totals):,.0f}")
        print(f"  最低周产能: {min(weekly_totals):,.0f}")
        
        # 如果成功率达到95%以上，记录为最少供应商数量
        if success_rate >= 0.95 and results['minimum_suppliers'] == 0:
            results['minimum_suppliers'] = num_suppliers
            print(f"★ 找到最少供应商数量: {num_suppliers} 家 (成功率: {success_rate:.2%})")
        
        # 如果连续3次成功率都很高，可以提前停止
        if (len(results['success_rates']) >= 3 and 
            all(rate >= 0.98 for rate in results['success_rates'][-3:])):
            print("连续达到高成功率，停止测试")
            break
    
    # 如果没有找到满足条件的最少数量，使用达到最高成功率的数量
    if results['minimum_suppliers'] == 0:
        max_success_idx = np.argmax(results['success_rates'])
        results['minimum_suppliers'] = max_success_idx + 1
        print(f"使用最高成功率对应的供应商数量: {results['minimum_suppliers']} 家")
    
    return results


def monte_carlo_fallback(supplier_selection, target_capacity, simulation_weeks):
    """传统蒙特卡洛模拟方法（备用）"""
    print("使用传统蒙特卡洛方法...")
    
    results = {
        'success_rates': [],
        'minimum_suppliers': 0,
        'detailed_results': []
    }
    
    suppliers = supplier_selection.copy()
    
    for num_suppliers in range(1, min(31, len(suppliers) + 1)):
        current_suppliers = suppliers.head(num_suppliers)
        
        successful_simulations = 0
        
        for _ in range(100):  # 简化的模拟次数
            weekly_totals = []
            
            for week in range(simulation_weeks):
                week_total = 0
                for _, supplier in current_suppliers.iterrows():
                    avg_capacity = supplier.get('平均周制造能力', 0)
                    # 添加随机波动
                    supply = avg_capacity * (0.7 + 0.6 * np.random.random())
                    week_total += supply
                weekly_totals.append(week_total)
            
            # 检查是否所有周都满足目标
            if all(total >= target_capacity for total in weekly_totals):
                successful_simulations += 1
        
        success_rate = successful_simulations / 100
        results['success_rates'].append(success_rate)
        
        if success_rate >= 0.95 and results['minimum_suppliers'] == 0:
            results['minimum_suppliers'] = num_suppliers
            break
    
    return results


# 快速测试函数
def test_ml_monte_carlo():
    """测试ML版本的蒙特卡洛模拟"""
    print("测试ML版本蒙特卡洛模拟...")
    
    # 创建测试数据
    test_suppliers = pd.DataFrame({
        '供应商ID': ['S229', 'S348', 'S016'],
        '材料分类': ['A', 'A', 'A'],
        '平均周制造能力': [1500, 700, 50]  # 基于前面的分析结果
    })
    
    # 运行测试
    results = monte_carlo_minimum_suppliers_ml(
        test_suppliers, 
        target_capacity=5000,  # 设置较低的目标便于测试
        simulation_weeks=4,
        num_simulations=100
    )
    
    print("\n测试结果:")
    for i, result in enumerate(results['detailed_results']):
        print(f"  {result['num_suppliers']}家供应商: 成功率{result['success_rate']:.2%}, "
              f"平均产能{result['avg_weekly_capacity']:.0f}")
    
    print(f"\n推荐最少供应商数量: {results['minimum_suppliers']} 家")


if __name__ == "__main__":
    test_ml_monte_carlo()
