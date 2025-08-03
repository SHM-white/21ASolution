"""
蒙特卡洛模拟计算最少供应商数量
基于ML预测模型的高精度模拟
"""

import pandas as pd
import numpy as np
from supplier_prediction_model_v2 import predict_multiple_suppliers, get_trained_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """蒙特卡洛模拟器"""
    
    def __init__(self):
        self.target_weekly_capacity = 28200  # 企业周产能需求（立方米）
        self.planning_weeks = 24  # 规划周数
        self.safety_margin = 1.1  # 安全边际 (10%)
        self.success_threshold = 0.90  # 成功率阈值 (90%)
        
        # 材料转换系数（原材料 -> 产品）
        self.material_conversion = {
            'A': 1/0.6,    # 1.6667
            'B': 1/0.66,   # 1.5152
            'C': 1/0.72    # 1.3889
        }
        
    def load_supplier_data(self):
        """加载供应商基础数据"""
        print("加载供应商基础数据...")
        
        # 1. 加载供应商产品制造能力汇总
        capacity_summary = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        print(f"制造能力数据: {capacity_summary.shape}")
        
        # 2. 加载供应商可靠性排名
        reliability_ranking = pd.read_excel('DataFrames/供应商可靠性年度加权排名.xlsx')
        print(f"可靠性排名数据: {reliability_ranking.shape}")
        
        # 3. 合并数据，创建供应商选择池
        supplier_pool = []
        
        for _, row in capacity_summary.iterrows():
            supplier_id = row['供应商ID']
            material_type = row['材料分类']
            avg_capacity = row['平均周制造能力']
            max_capacity = row['最大周制造能力']
            stability = row['制造能力稳定性']
            
            # 查找可靠性评级
            reliability_info = reliability_ranking[
                reliability_ranking['供应商名称'] == supplier_id
            ]
            
            if not reliability_info.empty:
                reliability_score = reliability_info.iloc[0].get('综合可靠性得分', 0.5)
                weight_ranking = reliability_info.iloc[0].get('加权排名', 999)
            else:
                reliability_score = 0.5
                weight_ranking = 999
            
            supplier_pool.append({
                'supplier_id': supplier_id,
                'material_type': material_type,
                'avg_weekly_capacity': avg_capacity,
                'max_weekly_capacity': max_capacity,
                'stability': stability,
                'reliability_score': reliability_score,
                'weight_ranking': weight_ranking,
                'conversion_factor': self.material_conversion[material_type]
            })
        
        supplier_df = pd.DataFrame(supplier_pool)
        
        # 按综合评分排序（可靠性 + 产能）
        supplier_df['composite_score'] = (
            supplier_df['reliability_score'] * 0.6 + 
            (supplier_df['avg_weekly_capacity'] / supplier_df['avg_weekly_capacity'].max()) * 0.4
        )
        
        supplier_df = supplier_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        print(f"供应商池构建完成: {len(supplier_df)} 家供应商")
        print(f"材料类型分布:")
        for material in ['A', 'B', 'C']:
            count = len(supplier_df[supplier_df['material_type'] == material])
            total_capacity = supplier_df[supplier_df['material_type'] == material]['avg_weekly_capacity'].sum()
            print(f"  {material}类: {count}家, 总产能: {total_capacity:.0f}")
        
        return supplier_df
    
    def simulate_supply_scenario(self, selected_suppliers, num_simulations=500, show_progress=True):
        """
        模拟供应场景
        
        参数:
        - selected_suppliers: 选定的供应商DataFrame
        - num_simulations: 模拟次数
        - show_progress: 是否显示进度信息
        
        返回:
        - 模拟结果字典
        """
        if show_progress:
            print(f"开始蒙特卡洛模拟...")
            print(f"  选定供应商数量: {len(selected_suppliers)}")
            print(f"  模拟次数: {num_simulations}")
            print(f"  目标周产能: {self.target_weekly_capacity:,} 立方米")
        
        # 获取供应商ID列表
        supplier_ids = selected_suppliers['supplier_id'].tolist()
        
        success_count = 0
        weekly_capacities_all = []
        min_weekly_capacities = []
        
        for sim in range(num_simulations):
            if show_progress and (sim + 1) % 100 == 0:
                print(f"  进度: {sim + 1}/{num_simulations}")
            
            try:
                # 使用ML模型预测每个供应商的供货量
                predictions = predict_multiple_suppliers(supplier_ids, self.planning_weeks, use_multithread=True)
                
                # 计算每周的总制造能力
                weekly_capacities = []
                
                for week in range(self.planning_weeks):
                    week_total_capacity = 0
                    
                    for _, supplier in selected_suppliers.iterrows():
                        supplier_id = supplier['supplier_id']
                        
                        if supplier_id in predictions:
                            # 原材料供货量
                            raw_supply = predictions[supplier_id][week]
                            
                            # 转换为产品制造能力
                            product_capacity = raw_supply * supplier['conversion_factor']
                            
                            # 考虑可靠性因子
                            reliability_factor = supplier['reliability_score']
                            actual_capacity = product_capacity * reliability_factor
                            
                            week_total_capacity += actual_capacity
                    
                    weekly_capacities.append(week_total_capacity)
                
                weekly_capacities_all.append(weekly_capacities)
                min_weekly = min(weekly_capacities)
                min_weekly_capacities.append(min_weekly)
                
                # 判断是否成功（最低周产能满足需求）
                if min_weekly >= self.target_weekly_capacity:
                    success_count += 1
            
            except Exception as e:
                print(f"  模拟 {sim+1} 失败: {e}")
                # 使用备选方法：基于历史平均值
                weekly_capacities = []
                for week in range(self.planning_weeks):
                    week_total = 0
                    for _, supplier in selected_suppliers.iterrows():
                        # 使用平均产能 + 随机波动
                        base_capacity = supplier['avg_weekly_capacity']
                        volatility = supplier['stability'] / base_capacity if base_capacity > 0 else 0.2
                        actual_capacity = base_capacity * (1 + np.random.normal(0, volatility))
                        actual_capacity = max(0, actual_capacity)  # 确保非负
                        week_total += actual_capacity * supplier['reliability_score']
                    weekly_capacities.append(week_total)
                
                weekly_capacities_all.append(weekly_capacities)
                min_weekly = min(weekly_capacities)
                min_weekly_capacities.append(min_weekly)
                
                if min_weekly >= self.target_weekly_capacity:
                    success_count += 1
        
        # 计算统计结果
        success_rate = success_count / num_simulations
        avg_min_capacity = np.mean(min_weekly_capacities)
        std_min_capacity = np.std(min_weekly_capacities)
        avg_all_weeks = np.mean([np.mean(weeks) for weeks in weekly_capacities_all])
        
        # 计算置信区间
        percentile_5 = np.percentile(min_weekly_capacities, 5)
        percentile_95 = np.percentile(min_weekly_capacities, 95)
        
        result = {
            'num_suppliers': len(selected_suppliers),
            'success_rate': success_rate,
            'avg_min_capacity': avg_min_capacity,
            'std_min_capacity': std_min_capacity,
            'avg_all_weeks_capacity': avg_all_weeks,
            'confidence_interval_5_95': (percentile_5, percentile_95),
            'target_capacity': self.target_weekly_capacity,
            'min_weekly_capacities': min_weekly_capacities,
            'weekly_capacities_all': weekly_capacities_all
        }
        
        if show_progress:
            print(f"  模拟完成!")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  平均最低周产能: {avg_min_capacity:,.0f}")
            print(f"  95%置信区间: [{percentile_5:,.0f}, {percentile_95:,.0f}]")
        
        return result
    
    def _test_supplier_count(self, num_suppliers, supplier_pool):
        """
        测试指定数量供应商的单个工作函数（用于多线程）
        
        参数:
        - num_suppliers: 供应商数量
        - supplier_pool: 供应商池DataFrame
        
        返回:
        - 模拟结果字典
        """
        try:
            # 选择Top N供应商
            selected_suppliers = supplier_pool.head(num_suppliers)
            
            # 进行蒙特卡洛模拟
            simulation_result = self.simulate_supply_scenario(
                selected_suppliers, 
                num_simulations=100, 
                show_progress=False  # 多线程时不显示内部进度
            )
            
            # 添加供应商组成信息
            material_counts = selected_suppliers['material_type'].value_counts()
            composition = {}
            for material in ['A', 'B', 'C']:
                count = material_counts.get(material, 0)
                if count > 0:
                    total_capacity = selected_suppliers[
                        selected_suppliers['material_type'] == material
                    ]['avg_weekly_capacity'].sum()
                    composition[material] = {
                        'count': count,
                        'total_capacity': total_capacity
                    }
            
            simulation_result['composition'] = composition
            return simulation_result
            
        except Exception as e:
            print(f"  ✗ 测试 {num_suppliers} 家供应商时出错: {e}")
            return None

    def find_minimum_suppliers(self, max_suppliers=402, step_size=5, use_multithread=True, start_count=200, max_workers=None):
        """
        寻找满足需求的最少供应商数量
        
        参数:
        - max_suppliers: 最大测试供应商数量
        - step_size: 步长
        - use_multithread: 是否使用多线程
        - max_workers: 最大工作线程数
        
        返回:
        - 结果字典
        """
        print("=" * 60)
        print("寻找满足需求的最少供应商数量")
        print("=" * 60)
        
        # 加载供应商数据
        supplier_pool = self.load_supplier_data()
        
        # 确保训练好了ML模型
        print("确保ML模型已训练...")
        try:
            model = get_trained_model()
            print("✓ ML模型已就绪")
        except Exception as e:
            print(f"✗ ML模型初始化失败: {e}")
            return None
        
        # 生成要测试的供应商数量列表
        test_counts = list(range(start_count, min(max_suppliers + 1, len(supplier_pool) + 1), step_size))
        print(f"将测试 {len(test_counts)} 种不同的供应商数量组合: {test_counts}")
        
        results = []
        recommended_count = None
        
        if use_multithread and len(test_counts) > 1:
            # 多线程并行测试
            if max_workers is None:
                max_workers = min(20, (os.cpu_count() or 1))  # 限制最大线程数，避免过度消耗资源
            
            print(f"🚀 使用多线程模式，最大线程数: {max_workers}")
            print("=" * 60)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_count = {
                    executor.submit(self._test_supplier_count, num_suppliers, supplier_pool): num_suppliers 
                    for num_suppliers in test_counts
                }
                
                # 使用tqdm显示总体进度
                with tqdm(total=len(test_counts), desc="测试不同供应商数量", unit="组合") as pbar:
                    for future in as_completed(future_to_count):
                        num_suppliers = future_to_count[future]
                        
                        try:
                            simulation_result = future.result()
                            
                            if simulation_result is not None:
                                results.append(simulation_result)
                                
                                # 更新进度条描述
                                success_rate = simulation_result['success_rate']
                                pbar.set_postfix({
                                    f'{num_suppliers}家': f'{success_rate:.1%}',
                                    '目标': f'{self.success_threshold:.0%}'
                                })
                                
                                # 检查是否达到成功率要求
                                if success_rate >= self.success_threshold and recommended_count is None:
                                    recommended_count = num_suppliers
                                    print(f"\n★ 找到推荐方案！{num_suppliers} 家供应商，成功率: {success_rate:.2%}")
                            
                        except Exception as e:
                            print(f"\n✗ 测试 {num_suppliers} 家供应商时出错: {e}")
                        
                        pbar.update(1)
            
            # 按供应商数量排序结果
            results.sort(key=lambda x: x['num_suppliers'])
            
        else:
            # 单线程顺序测试（原有逻辑）
            print("🔄 使用单线程模式")
            print("=" * 60)
            
            for num_suppliers in test_counts:
                print(f"\n--- 测试 {num_suppliers} 家供应商 ---")
                
                # 选择Top N供应商
                selected_suppliers = supplier_pool.head(num_suppliers)
                
                # 显示选择的供应商组合
                material_counts = selected_suppliers['material_type'].value_counts()
                print(f"选择的供应商构成:")
                for material in ['A', 'B', 'C']:
                    count = material_counts.get(material, 0)
                    if count > 0:
                        avg_capacity = selected_suppliers[selected_suppliers['material_type'] == material]['avg_weekly_capacity'].sum()
                        print(f"  {material}类: {count}家, 总产能: {avg_capacity:,.0f}")
                
                # 进行蒙特卡洛模拟
                simulation_result = self.simulate_supply_scenario(selected_suppliers, num_simulations=100)
                
                results.append(simulation_result)
                
                # 判断是否达到成功率要求
                if simulation_result['success_rate'] >= self.success_threshold and recommended_count is None:
                    recommended_count = num_suppliers
                    print(f"★ 找到推荐方案！")
                    print(f"  推荐供应商数量: {num_suppliers} 家")
                    print(f"  成功率: {simulation_result['success_rate']:.2%}")
                    print(f"  平均最低周产能: {simulation_result['avg_min_capacity']:,.0f}")
                    
                    # 继续测试几个更大的组合以验证稳定性
                    if num_suppliers < max_suppliers - 2 * step_size:
                        print(f"  继续验证更大规模组合的稳定性...")
                        continue
                    else:
                        break
        
        # 汇总结果
        final_result = {
            'recommended_supplier_count': recommended_count,
            'simulation_results': results,
            'target_capacity': self.target_weekly_capacity,
            'success_threshold': self.success_threshold,
            'safety_margin': self.safety_margin,
            'planning_weeks': self.planning_weeks
        }
        
        print(f"\n" + "=" * 60)
        print(f"最终分析结果")
        print(f"=" * 60)
        
        if recommended_count:
            print(f"✓ 推荐最少供应商数量: {recommended_count} 家")
            
            # 找到推荐方案的详细结果
            recommended_result = None
            for result in results:
                if result['num_suppliers'] == recommended_count:
                    recommended_result = result
                    break
            
            if recommended_result:
                print(f"推荐方案详细信息:")
                print(f"  成功率: {recommended_result['success_rate']:.2%}")
                print(f"  平均最低周产能: {recommended_result['avg_min_capacity']:,.0f} 立方米")
                print(f"  目标周产能: {self.target_weekly_capacity:,} 立方米")
                print(f"  安全边际: {self.safety_margin:.1%}")
                print(f"  95%置信区间: [{recommended_result['confidence_interval_5_95'][0]:,.0f}, {recommended_result['confidence_interval_5_95'][1]:,.0f}]")
                
                # 显示供应商组成（如果有的话）
                if 'composition' in recommended_result:
                    print(f"  供应商组成:")
                    for material, info in recommended_result['composition'].items():
                        print(f"    {material}类: {info['count']}家, 总产能: {info['total_capacity']:,.0f}")
                else:
                    # 多线程模式可能没有composition信息，手动计算
                    selected_suppliers = supplier_pool.head(recommended_count)
                    material_counts = selected_suppliers['material_type'].value_counts()
                    print(f"  供应商组成:")
                    for material in ['A', 'B', 'C']:
                        count = material_counts.get(material, 0)
                        if count > 0:
                            total_capacity = selected_suppliers[selected_suppliers['material_type'] == material]['avg_weekly_capacity'].sum()
                            print(f"    {material}类: {count}家, 总产能: {total_capacity:,.0f}")
        else:
            print(f"✗ 在测试范围内未找到满足 {self.success_threshold:.0%} 成功率的方案")
            print(f"建议:")
            print(f"  1. 增加测试的供应商数量上限")
            print(f"  2. 降低成功率要求")
            print(f"  3. 增加安全边际")
        
        print(f"\n所有测试结果汇总:")
        for result in results:
            print(f"  {result['num_suppliers']:3d}家供应商: 成功率 {result['success_rate']:6.2%}, "
                  f"平均最低产能 {result['avg_min_capacity']:8,.0f}")
        
        return final_result


def main():
    """主函数"""
    print("=" * 80)
    print("蒙特卡洛模拟 - 最少供应商数量计算")
    print("基于ML预测模型的高精度分析")
    print("=" * 80)
    
    # 创建模拟器
    simulator = MonteCarloSimulator()
    
    # 设置参数
    print(f"模拟参数:")
    print(f"  目标周产能: {simulator.target_weekly_capacity:,} 立方米")
    print(f"  规划周数: {simulator.planning_weeks}")
    print(f"  成功率要求: {simulator.success_threshold:.0%}")
    print(f"  安全边际: {simulator.safety_margin:.1%}")
    
    # 执行分析
    try:
        result = simulator.find_minimum_suppliers(
            max_suppliers=402, 
            step_size=10, 
            use_multithread=True,
            start_count=300,
            max_workers=20  # 限制线程数，避免过度消耗资源
        )
        
        if result:
            print(f"\n分析成功完成!")
            return result
        else:
            print(f"\n分析失败!")
            return None
            
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 运行主分析
    result = main()
    
    if result:
        print(f"\n=" * 60)
        print(f"程序执行完成")
        print(f"=" * 60)
        
        if result['recommended_supplier_count']:
            print(f"推荐结果: 至少需要 {result['recommended_supplier_count']} 家供应商")
        else:
            print(f"需要进一步调整参数或增加供应商候选池")
    else:
        print(f"\n程序执行失败，请检查数据文件和模型配置")
