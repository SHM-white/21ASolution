"""
问题三：优化订购方案与转运方案
目标：多采购A类材料，少采购C类材料，减少转运及仓储成本，降低转运损耗率

策略:
1. 优先选择A类高评分供应商，降低C类采购比例
2. 使用EOQ模型优化B类采购量
3. 选择低损耗率转运商，A类材料优先配置最优转运商
4. 满足24周生产需求和两周安全库存约束
"""

import pandas as pd
import numpy as np
import warnings
from math import sqrt
from scipy.optimize import minimize
from itertools import combinations
import logging
import os
from datetime import datetime
from tqdm import tqdm
from supplier_prediction_model_v3 import predict_multiple_suppliers, get_trained_timeseries_model
import copy

warnings.filterwarnings('ignore')

# 日志设置
os.makedirs('log', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'log/problem3_optimization_{timestamp}.log'
logging.basicConfig(filename=log_file, level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class Problem3Optimizer:
    """问题三优化器"""
    
    def __init__(self):
        # 生产参数
        self.weekly_capacity = 28200  # 周产能需求（立方米）
        self.planning_weeks = 24     # 规划周数
        self.safety_weeks = 2        # 安全库存周数
        
        # 原材料转换比例 (每立方米产品需要的原材料)
        self.material_conversion = {
            'A': 0.6,   # A类原材料转换比例
            'B': 0.66,  # B类原材料转换比例  
            'C': 0.72   # C类原材料转换比例
        }
        
        # 原材料相对价格 (以C类为基准1.0)
        self.material_prices = {
            'A': 1.2,   # A类比C类高20%
            'B': 1.1,   # B类比C类高10%
            'C': 1.0    # C类基准价格
        }
        
        # 转运商运输能力
        self.transporter_capacity = 6000  # 立方米/周
        
        # 优化权重
        self.weights = {
            'cost_reduction': 0.4,      # 成本降低权重
            'loss_minimization': 0.3,   # 损耗最小化权重
            'a_maximization': 0.2,      # A类最大化权重
            'c_minimization': 0.1       # C类最小化权重
        }
        
        # EOQ模型参数
        self.eoq_params = {
            'ordering_cost': 1000,      # 订货成本
            'holding_cost_rate': 0.2    # 仓储成本率
        }
        
        self.supplier_data = None
        self.transporter_data = None
        self.optimal_suppliers = None
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 1. 加载供应商制造能力数据
        capacity_df = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        
        # 2. 加载供应商可靠性排名（Top 50）
        reliability_df = pd.read_excel('DataFrames/供应商可靠性年度加权排名.xlsx')
        
        # 3. 加载转运商数据
        transporter_df = pd.read_excel('DataFrames/转运商损耗率分析结果.xlsx')
        
        # 4. 合并供应商数据
        self.supplier_data = []
        
        for _, row in capacity_df.iterrows():
            supplier_id = row['供应商ID']
            material_type = row['材料分类']
            avg_capacity = row['平均周制造能力']
            max_capacity = row['最大周制造能力']
            stability = row['制造能力稳定性']
            
            # 查找可靠性信息
            reliability_info = reliability_df[
                reliability_df['供应商名称'] == supplier_id
            ]
            
            if not reliability_info.empty:
                reliability_score = reliability_info.iloc[0]['加权可靠性得分']
                ranking = reliability_info.iloc[0]['排名']
                is_top50 = True
            else:
                # 如果不在Top50中，给一个较低的评分
                reliability_score = 15.0  # 较低评分
                ranking = 999
                is_top50 = False
            
            self.supplier_data.append({
                'supplier_id': supplier_id,
                'material_type': material_type,
                'avg_weekly_capacity': avg_capacity,
                'max_weekly_capacity': max_capacity,
                'stability': stability,
                'reliability_score': reliability_score,
                'ranking': ranking,
                'is_top50': is_top50,
                'conversion_factor': 1 / self.material_conversion[material_type]
            })
        
        self.supplier_data = pd.DataFrame(self.supplier_data)
        
        # 5. 转运商数据
        self.transporter_data = transporter_df[['transporter_name', 'avg_loss_rate', 
                                               'stability_score', 'comprehensive_score']].copy()
        
        print(f"✓ 加载完成：{len(self.supplier_data)} 家供应商，{len(self.transporter_data)} 家转运商")
        
        # 显示各类材料供应商数量
        for material in ['A', 'B', 'C']:
            count = len(self.supplier_data[self.supplier_data['material_type'] == material])
            top50_count = len(self.supplier_data[
                (self.supplier_data['material_type'] == material) & 
                (self.supplier_data['is_top50'] == True)
            ])
            print(f"  {material}类供应商：{count}家 (Top50: {top50_count}家)")
    
    def classify_suppliers(self):
        """供应商分类和筛选"""
        print("\n供应商分类和筛选...")
        
        # 按材料类型和评分分组
        # A类：优先选择Top50中的高评分供应商，然后扩展到其他A类供应商
        group_A = self.supplier_data[
            (self.supplier_data['material_type'] == 'A') & 
            (self.supplier_data['reliability_score'] > 20)  # 较高评分阈值
        ].sort_values('reliability_score', ascending=False)
        
        # B类：选择中等评分以上的供应商
        group_B = self.supplier_data[
            (self.supplier_data['material_type'] == 'B') & 
            (self.supplier_data['reliability_score'] > 15)  # 中等评分阈值
        ].sort_values('reliability_score', ascending=False)
        
        # C类：仅选择必要的供应商，评分可以较低
        group_C = self.supplier_data[
            (self.supplier_data['material_type'] == 'C') & 
            (self.supplier_data['reliability_score'] > 10)  # 较低评分阈值
        ].sort_values('reliability_score', ascending=False)
        
        print(f"  分类结果：A类 {len(group_A)} 家，B类 {len(group_B)} 家，C类 {len(group_C)} 家")
        
        return group_A, group_B, group_C
    
    def calculate_eoq(self, demand, material_type):
        """计算EOQ最优订货量"""
        # EOQ = sqrt(2 * D * S / H)
        # D: 需求量, S: 订货成本, H: 仓储成本
        holding_cost = self.eoq_params['holding_cost_rate'] * self.material_prices[material_type]
        
        eoq = sqrt(2 * demand * self.eoq_params['ordering_cost'] / holding_cost)
        return eoq
    
    def strategy1_base_adjustment(self, group_A, group_B, group_C):
        """
        策略1：基于问题二结果的增量调整
        A类追加，C类削减，B类维持
        """
        print("\n执行策略1：基于增量调整...")
        
        # 目标：A类占40%以上，C类占25%以下，B类占35%左右
        target_A_ratio = 0.45
        target_B_ratio = 0.35  
        target_C_ratio = 0.20
        
        weekly_orders = []
        
        for week in range(self.planning_weeks):
            week_order = {}
            week_total = 0
            
            # 计算本周需求（含安全库存）
            base_demand = self.weekly_capacity
            if week < self.safety_weeks:
                safety_demand = base_demand * self.safety_weeks
                total_demand = base_demand + safety_demand
            else:
                total_demand = base_demand
            
            # A类供应商选择（从高评分开始）
            A_target = total_demand * target_A_ratio
            A_allocated = 0
            
            for _, supplier in group_A.head(50).iterrows():  # 扩展到50家最优A类供应商
                if A_allocated >= A_target:
                    break
                
                # 使用预测模型预测供货能力
                predicted_capacity = self._predict_supplier_capacity(
                    supplier['supplier_id'], week
                )
                
                # 分配订货量（不超过供应商最大能力的80%）
                max_order = min(
                    predicted_capacity * 0.8,
                    supplier['max_weekly_capacity'] * 0.8,
                    A_target - A_allocated
                )
                
                if max_order > 0:
                    week_order[supplier['supplier_id']] = max_order
                    A_allocated += max_order
                    week_total += max_order
            
            # B类供应商选择（使用EOQ模型）
            B_target = total_demand * target_B_ratio
            B_allocated = 0
            eoq_B = self.calculate_eoq(B_target, 'B')
            
            for _, supplier in group_B.head(30).iterrows():  # 选择30家最优B类供应商
                if B_allocated >= B_target:
                    break
                
                predicted_capacity = self._predict_supplier_capacity(
                    supplier['supplier_id'], week
                )
                
                # EOQ约束下的分配
                max_order = min(
                    predicted_capacity * 0.8,
                    eoq_B / len(group_B.head(30)),  # 平均分配EOQ量
                    B_target - B_allocated
                )
                
                if max_order > 0:
                    week_order[supplier['supplier_id']] = max_order
                    B_allocated += max_order
                    week_total += max_order
            
            # C类供应商选择（最少化）
            remaining_demand = max(0, total_demand - week_total)
            C_allocated = 0
            
            for _, supplier in group_C.head(15).iterrows():  # 仅选择15家最优C类供应商
                if remaining_demand <= 0:
                    break
                
                predicted_capacity = self._predict_supplier_capacity(
                    supplier['supplier_id'], week
                )
                
                max_order = min(
                    predicted_capacity * 0.8,
                    remaining_demand
                )
                
                if max_order > 0:
                    week_order[supplier['supplier_id']] = max_order
                    C_allocated += max_order
                    remaining_demand -= max_order
            
            weekly_orders.append(week_order)
            
            # 记录本周分配情况
            actual_A_ratio = A_allocated / total_demand if total_demand > 0 else 0
            actual_B_ratio = B_allocated / total_demand if total_demand > 0 else 0
            actual_C_ratio = C_allocated / total_demand if total_demand > 0 else 0
            
            if week % 5 == 0:  # 每5周输出一次进度
                print(f"  第{week+1}周：A类{actual_A_ratio:.1%}，B类{actual_B_ratio:.1%}，C类{actual_C_ratio:.1%}")
        
        return weekly_orders
    
    def strategy2_priority_driven(self, group_A, group_B, group_C):
        """
        策略2：优先级驱动策略
        A类优先 + B类EOQ + C类补充
        """
        print("\n执行策略2：优先级驱动...")
        
        weekly_orders = []
        
        for week in range(self.planning_weeks):
            week_order = {}
            
            # 计算本周需求
            base_demand = self.weekly_capacity
            if week < self.safety_weeks:
                safety_demand = base_demand * self.safety_weeks
                total_demand = base_demand + safety_demand
            else:
                total_demand = base_demand
            
            # 阶段1：A类优先（目标覆盖50%产能）
            A_target = total_demand * 0.5
            A_allocated = 0
            
            for _, supplier in group_A.head(60).iterrows():  # 扩展A类供应商池
                if A_allocated >= A_target:
                    break
                
                predicted_capacity = self._predict_supplier_capacity(
                    supplier['supplier_id'], week
                )
                
                max_order = min(
                    predicted_capacity * 0.85,  # 更积极的分配
                    A_target - A_allocated
                )
                
                if max_order > 0:
                    week_order[supplier['supplier_id']] = max_order
                    A_allocated += max_order
            
            # 阶段2：B类EOQ模型补充
            remaining_demand = total_demand - A_allocated
            B_demand = max(0, remaining_demand * 0.7)  # B类承担剩余需求的70%
            B_allocated = 0
            
            if B_demand > 0:
                eoq_B = self.calculate_eoq(B_demand, 'B')
                suppliers_count = min(25, len(group_B))
                
                for _, supplier in group_B.head(suppliers_count).iterrows():
                    if B_allocated >= B_demand:
                        break
                    
                    predicted_capacity = self._predict_supplier_capacity(
                        supplier['supplier_id'], week
                    )
                    
                    # 单供应商不超过EOQ推荐量的1/n
                    max_order = min(
                        predicted_capacity * 0.8,
                        eoq_B / suppliers_count,
                        B_demand - B_allocated
                    )
                    
                    if max_order > 0:
                        week_order[supplier['supplier_id']] = max_order
                        B_allocated += max_order
            
            # 阶段3：C类仅填补缺口
            final_remaining = max(0, total_demand - A_allocated - B_allocated)
            C_allocated = 0
            
            if final_remaining > 0:
                for _, supplier in group_C.head(10).iterrows():  # 最少化C类供应商
                    if final_remaining <= 0:
                        break
                    
                    predicted_capacity = self._predict_supplier_capacity(
                        supplier['supplier_id'], week
                    )
                    
                    max_order = min(
                        predicted_capacity * 0.7,  # 保守分配
                        final_remaining
                    )
                    
                    if max_order > 0:
                        week_order[supplier['supplier_id']] = max_order
                        C_allocated += max_order
                        final_remaining -= max_order
            
            weekly_orders.append(week_order)
            
            # 进度输出
            if week % 5 == 0:
                total_allocated = A_allocated + B_allocated + C_allocated
                print(f"  第{week+1}周：总需求{total_demand:.0f}，已分配{total_allocated:.0f}")
        
        return weekly_orders
    
    def _predict_supplier_capacity(self, supplier_id, week):
        """预测供应商在特定周的供货能力"""
        try:
            # 使用现有的预测模型
            predictions = predict_multiple_suppliers([supplier_id], 1, use_multithread=False)
            
            if supplier_id in predictions and len(predictions[supplier_id]) > 0:
                return max(0, predictions[supplier_id][0])
            else:
                # 备选：使用历史平均值
                supplier_info = self.supplier_data[
                    self.supplier_data['supplier_id'] == supplier_id
                ]
                if not supplier_info.empty:
                    return supplier_info.iloc[0]['avg_weekly_capacity']
                else:
                    return 0
        except:
            # 异常处理：使用历史平均值
            supplier_info = self.supplier_data[
                self.supplier_data['supplier_id'] == supplier_id
            ]
            if not supplier_info.empty:
                return supplier_info.iloc[0]['avg_weekly_capacity']
            else:
                return 0
    
    def optimize_transportation(self, weekly_orders):
        """优化转运方案"""
        print("\n优化转运方案...")
        
        # 转运商按综合评分排序（损耗率低、稳定性高）
        transporters = self.transporter_data.sort_values('comprehensive_score', ascending=False)
        
        weekly_transport_plans = []
        
        for week, orders in enumerate(weekly_orders):
            transport_plan = {}
            
            # 按材料类型分组订单
            material_orders = {'A': {}, 'B': {}, 'C': {}}
            
            for supplier_id, amount in orders.items():
                supplier_info = self.supplier_data[
                    self.supplier_data['supplier_id'] == supplier_id
                ]
                if not supplier_info.empty:
                    material_type = supplier_info.iloc[0]['material_type']
                    material_orders[material_type][supplier_id] = amount
            
            # 为每种材料分配转运商
            # A类材料：优先使用最优转运商
            self._assign_transporters(material_orders['A'], transporters, 'A', transport_plan, week)
            
            # B类材料：使用中等转运商
            self._assign_transporters(material_orders['B'], transporters, 'B', transport_plan, week)
            
            # C类材料：使用剩余转运商
            self._assign_transporters(material_orders['C'], transporters, 'C', transport_plan, week)
            
            weekly_transport_plans.append(transport_plan)
        
        return weekly_transport_plans
    
    def _assign_transporters(self, material_orders, transporters, material_type, transport_plan, week):
        """为特定材料类型分配转运商"""
        
        # 根据材料类型选择转运商优先级
        if material_type == 'A':
            # A类使用最优转运商（前3名）
            selected_transporters = transporters.head(3)
        elif material_type == 'B':
            # B类使用中等转运商（第2-5名）
            selected_transporters = transporters.iloc[1:5]
        else:  # C类
            # C类使用所有可用转运商
            selected_transporters = transporters
        
        # 转运商容量记录
        transporter_used_capacity = {t: 0 for t in selected_transporters['transporter_name']}
        
        # 为每个供应商分配转运商
        for supplier_id, amount in material_orders.items():
            allocated_amount = 0
            
            # 尝试用单个转运商运输
            for _, transporter in selected_transporters.iterrows():
                transporter_name = transporter['transporter_name']
                available_capacity = self.transporter_capacity - transporter_used_capacity[transporter_name]
                
                if available_capacity >= amount:
                    # 单个转运商可以完全承担
                    transport_plan[f"{supplier_id}_{transporter_name}"] = amount
                    transporter_used_capacity[transporter_name] += amount
                    allocated_amount = amount
                    break
            
            # 如果单个转运商无法承担，使用多个转运商
            if allocated_amount < amount:
                remaining_amount = amount - allocated_amount
                
                for _, transporter in selected_transporters.iterrows():
                    if remaining_amount <= 0:
                        break
                    
                    transporter_name = transporter['transporter_name']
                    available_capacity = self.transporter_capacity - transporter_used_capacity[transporter_name]
                    
                    if available_capacity > 0:
                        alloc_amount = min(available_capacity, remaining_amount)
                        key = f"{supplier_id}_{transporter_name}"
                        if key in transport_plan:
                            transport_plan[key] += alloc_amount
                        else:
                            transport_plan[key] = alloc_amount
                        
                        transporter_used_capacity[transporter_name] += alloc_amount
                        remaining_amount -= alloc_amount
    
    def evaluate_solution(self, weekly_orders, transport_plans):
        """评估解决方案"""
        print("\n评估解决方案...")
        
        total_cost = 0
        total_loss = 0
        material_stats = {'A': 0, 'B': 0, 'C': 0}
        
        for week, (orders, transport) in enumerate(zip(weekly_orders, transport_plans)):
            week_cost = 0
            week_loss = 0
            
            # 计算订购成本
            for supplier_id, amount in orders.items():
                supplier_info = self.supplier_data[
                    self.supplier_data['supplier_id'] == supplier_id
                ]
                if not supplier_info.empty:
                    material_type = supplier_info.iloc[0]['material_type']
                    cost = amount * self.material_prices[material_type]
                    week_cost += cost
                    material_stats[material_type] += amount
            
            # 计算转运损耗
            for transport_key, amount in transport.items():
                if '_' in transport_key:
                    supplier_id, transporter_name = transport_key.rsplit('_', 1)
                    
                    # 查找转运商损耗率
                    transporter_info = self.transporter_data[
                        self.transporter_data['transporter_name'] == transporter_name
                    ]
                    if not transporter_info.empty:
                        loss_rate = transporter_info.iloc[0]['avg_loss_rate'] / 100
                        loss_amount = amount * loss_rate
                        week_loss += loss_amount
            
            total_cost += week_cost
            total_loss += week_loss
        
        # 计算材料比例
        total_material = sum(material_stats.values())
        material_ratios = {k: v/total_material if total_material > 0 else 0 
                          for k, v in material_stats.items()}
        
        # 综合评估
        evaluation = {
            'total_cost': total_cost,
            'total_loss': total_loss,
            'material_stats': material_stats,
            'material_ratios': material_ratios,
            'a_ratio': material_ratios['A'],
            'b_ratio': material_ratios['B'], 
            'c_ratio': material_ratios['C'],
            'cost_per_unit': total_cost / (self.weekly_capacity * self.planning_weeks),
            'loss_rate': total_loss / total_material if total_material > 0 else 0
        }
        
        return evaluation
    
    def save_results(self, strategy_name, weekly_orders, transport_plans, evaluation):
        """保存结果"""
        print(f"\n保存{strategy_name}结果...")
        
        # 保存订购方案
        order_df = []
        for week, orders in enumerate(weekly_orders):
            for supplier_id, amount in orders.items():
                order_df.append({
                    'week': week + 1,
                    'supplier_id': supplier_id,
                    'order_amount': amount
                })
        
        order_df = pd.DataFrame(order_df)
        order_file = f'results/problem3_{strategy_name}_orders.xlsx'
        os.makedirs('results', exist_ok=True)
        order_df.to_excel(order_file, index=False)
        
        # 保存转运方案
        transport_df = []
        for week, transport in enumerate(transport_plans):
            for transport_key, amount in transport.items():
                if '_' in transport_key:
                    supplier_id, transporter_name = transport_key.rsplit('_', 1)
                    transport_df.append({
                        'week': week + 1,
                        'supplier_id': supplier_id,
                        'transporter_name': transporter_name,
                        'transport_amount': amount
                    })
        
        transport_df = pd.DataFrame(transport_df)
        transport_file = f'results/problem3_{strategy_name}_transport.xlsx'
        transport_df.to_excel(transport_file, index=False)
        
        # 保存评估结果
        eval_file = f'results/problem3_{strategy_name}_evaluation.txt'
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write(f"{strategy_name}方案评估结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"总成本: {evaluation['total_cost']:,.2f}\n")
            f.write(f"总损耗: {evaluation['total_loss']:,.2f}\n")
            f.write(f"单位成本: {evaluation['cost_per_unit']:.4f}\n")
            f.write(f"损耗率: {evaluation['loss_rate']:.2%}\n")
            f.write(f"A类比例: {evaluation['a_ratio']:.2%}\n")
            f.write(f"B类比例: {evaluation['b_ratio']:.2%}\n")
            f.write(f"C类比例: {evaluation['c_ratio']:.2%}\n")
            f.write(f"A类总量: {evaluation['material_stats']['A']:,.0f}\n")
            f.write(f"B类总量: {evaluation['material_stats']['B']:,.0f}\n")
            f.write(f"C类总量: {evaluation['material_stats']['C']:,.0f}\n")
        
        print(f"  ✓ 结果已保存到 results/ 目录")
    
    def run_optimization(self):
        """运行完整优化流程"""
        print("=" * 60)
        print("问题三：订购方案与转运方案优化")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 供应商分类
        group_A, group_B, group_C = self.classify_suppliers()
        
        # 3. 确保预测模型已训练
        try:
            model = get_trained_timeseries_model()
            print("✓ 预测模型已就绪")
        except Exception as e:
            print(f"⚠ 预测模型初始化失败，将使用历史平均值: {e}")
        
        strategies = []
        
        # 4. 策略1：增量调整
        print("\n" + "=" * 40)
        print("执行策略1：基于增量调整")
        print("=" * 40)
        
        orders_1 = self.strategy1_base_adjustment(group_A, group_B, group_C)
        transport_1 = self.optimize_transportation(orders_1)
        eval_1 = self.evaluate_solution(orders_1, transport_1)
        self.save_results("strategy1", orders_1, transport_1, eval_1)
        strategies.append(("策略1", eval_1))
        
        # 5. 策略2：优先级驱动
        print("\n" + "=" * 40)
        print("执行策略2：优先级驱动")
        print("=" * 40)
        
        orders_2 = self.strategy2_priority_driven(group_A, group_B, group_C)
        transport_2 = self.optimize_transportation(orders_2)
        eval_2 = self.evaluate_solution(orders_2, transport_2)
        self.save_results("strategy2", orders_2, transport_2, eval_2)
        strategies.append(("策略2", eval_2))
        
        # 6. 结果对比
        print("\n" + "=" * 60)
        print("策略对比结果")
        print("=" * 60)
        
        for strategy_name, evaluation in strategies:
            print(f"\n{strategy_name}:")
            print(f"  A类比例: {evaluation['a_ratio']:.2%} (目标: 最大化)")
            print(f"  C类比例: {evaluation['c_ratio']:.2%} (目标: 最小化)")
            print(f"  损耗率: {evaluation['loss_rate']:.2%} (目标: 最小化)")
            print(f"  单位成本: {evaluation['cost_per_unit']:.4f} (目标: 降低)")
        
        # 7. 推荐最优策略
        best_strategy = min(strategies, 
                           key=lambda x: x[1]['cost_per_unit'] + x[1]['loss_rate'] - x[1]['a_ratio'])
        
        print(f"\n推荐策略: {best_strategy[0]}")
        print(f"优势: A类比例{best_strategy[1]['a_ratio']:.1%}，C类比例{best_strategy[1]['c_ratio']:.1%}，损耗率{best_strategy[1]['loss_rate']:.2%}")
        
        return strategies


def main():
    """主函数"""
    optimizer = Problem3Optimizer()
    results = optimizer.run_optimization()
    
    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
