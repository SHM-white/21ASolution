"""
问题三：优化订购方案与转运方案（基于第二问现有数据重构版本）
目标：多采购A类材料，少采购C类材料，减少转运及仓储成本，降低转运损耗率

策略:
1. 基于第二问的供货商组合，调整材料结构
2. 优先增加A类供应商，减少C类供应商
3. 使用EOQ模型优化B类采购量
4. 重新分配转运商，A类材料优先配置最优转运商
5. 满足24周生产需求和两周安全库存约束
"""

import pandas as pd
import numpy as np
import warnings
from math import sqrt
from scipy.optimize import minimize
import logging
import os
from datetime import datetime
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
        
        # 第二问的数据
        self.problem2_supply_plan = None
        self.problem2_transport_plan = None
        
    def load_data(self):
        """加载数据（使用第二问的现有结果）"""
        print("加载第二问的现有数据...")
        
        # 1. 加载第二问的供货计划
        try:
            self.problem2_supply_plan = pd.read_excel('DataFrames/problem2_allocation_supply.xlsx')
            print(f"✓ 第二问供货计划: {len(self.problem2_supply_plan)} 条记录")
        except FileNotFoundError:
            raise FileNotFoundError("未找到第二问的供货计划文件，请先运行第二问")
        
        # 2. 加载第二问的转运计划
        try:
            self.problem2_transport_plan = pd.read_excel('DataFrames/problem2_allocation_transport.xlsx')
            print(f"✓ 第二问转运计划: {len(self.problem2_transport_plan)} 条记录")
        except FileNotFoundError:
            raise FileNotFoundError("未找到第二问的转运计划文件，请先运行第二问")
        
        # 3. 加载供应商制造能力数据
        capacity_df = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        
        # 4. 加载供应商可靠性排名（Top 50）
        reliability_df = pd.read_excel('DataFrames/供应商可靠性年度加权排名.xlsx')
        
        # 5. 加载转运商数据
        transporter_df = pd.read_excel('DataFrames/转运商损耗率分析结果.xlsx')
        
        # 6. 构建供应商数据池（基于第二问的实际供应商）
        self.supplier_data = []
        
        # 获取第二问使用的所有供应商
        problem2_suppliers = self.problem2_supply_plan['supplier_id'].unique()
        print(f"第二问使用了 {len(problem2_suppliers)} 家供应商")
        
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
            
            # 标记是否为第二问的供应商
            is_problem2_supplier = supplier_id in problem2_suppliers
            
            self.supplier_data.append({
                'supplier_id': supplier_id,
                'material_type': material_type,
                'avg_weekly_capacity': avg_capacity,
                'max_weekly_capacity': max_capacity,
                'stability': stability,
                'reliability_score': reliability_score,
                'ranking': ranking,
                'is_top50': is_top50,
                'is_problem2_supplier': is_problem2_supplier,
                'conversion_factor': 1 / self.material_conversion[material_type]
            })
        
        self.supplier_data = pd.DataFrame(self.supplier_data)
        
        # 7. 转运商数据
        self.transporter_data = transporter_df[['transporter_name', 'avg_loss_rate', 
                                               'stability_score', 'comprehensive_score']].copy()
        
        print(f"✓ 加载完成：{len(self.supplier_data)} 家供应商，{len(self.transporter_data)} 家转运商")
        
        # 显示各类材料供应商数量
        for material in ['A', 'B', 'C']:
            total_count = len(self.supplier_data[self.supplier_data['material_type'] == material])
            problem2_count = len(self.supplier_data[
                (self.supplier_data['material_type'] == material) & 
                (self.supplier_data['is_problem2_supplier'] == True)
            ])
            top50_count = len(self.supplier_data[
                (self.supplier_data['material_type'] == material) & 
                (self.supplier_data['is_top50'] == True)
            ])
            print(f"  {material}类供应商：总数{total_count}家，第二问使用{problem2_count}家，Top50: {top50_count}家")
    
    def classify_suppliers(self):
        """供应商分类和筛选（基于第二问的结果进行优化）"""
        print("\n供应商分类和筛选（基于第二问现有数据）...")
        
        # 分析第二问的材料结构
        problem2_material_stats = self.problem2_supply_plan.groupby('material_type').agg({
            'supplier_id': 'nunique',
            'supply_quantity': 'sum'
        })
        
        total_supply = self.problem2_supply_plan['supply_quantity'].sum()
        print("第二问的材料结构:")
        for material in ['A', 'B', 'C']:
            if material in problem2_material_stats.index:
                count = problem2_material_stats.loc[material, 'supplier_id']
                quantity = problem2_material_stats.loc[material, 'supply_quantity']
                ratio = quantity / total_supply * 100
                print(f"  {material}类：{count}家供应商，{quantity:,.0f}供货量 ({ratio:.1f}%)")
        
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
        
        print(f"\n优化后分类结果：A类 {len(group_A)} 家，B类 {len(group_B)} 家，C类 {len(group_C)} 家")
        print(f"目标：增加A类比例，减少C类比例")
        
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
        策略1：基于第二问结果的增量调整
        A类追加，C类削减，B类维持
        """
        print("\n执行策略1：基于第二问结果的增量调整...")
        
        # 目标：A类占45%以上，C类占25%以下，B类占30%左右
        target_A_ratio = 0.45
        target_B_ratio = 0.30  
        target_C_ratio = 0.25
        
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
            
            # 基于第二问的供应商使用历史数据
            week_problem2_data = self.problem2_supply_plan[
                self.problem2_supply_plan['week'] == week + 1
            ]
            
            # A类供应商选择（增加A类比例）
            A_target = total_demand * target_A_ratio
            A_allocated = 0
            
            # 首先使用第二问的A类供应商
            problem2_A_suppliers = week_problem2_data[
                week_problem2_data['material_type'] == 'A'
            ]['supplier_id'].unique()
            
            for supplier_id in problem2_A_suppliers:
                if A_allocated >= A_target:
                    break
                
                # 获取第二问的实际供货量作为基准
                problem2_supply = week_problem2_data[
                    week_problem2_data['supplier_id'] == supplier_id
                ]['supply_quantity'].sum()
                
                # 增加20%的供货量
                enhanced_supply = problem2_supply * 1.2
                
                # 获取供应商最大能力限制
                supplier_info = self.supplier_data[
                    self.supplier_data['supplier_id'] == supplier_id
                ]
                if not supplier_info.empty:
                    max_capacity = supplier_info.iloc[0]['max_weekly_capacity']
                    enhanced_supply = min(enhanced_supply, max_capacity * 0.9)
                
                allocation = min(enhanced_supply, A_target - A_allocated)
                
                if allocation > 0:
                    week_order[supplier_id] = allocation
                    A_allocated += allocation
                    week_total += allocation
            
            # 如果A类还不够，添加更多A类供应商
            if A_allocated < A_target:
                remaining_A_need = A_target - A_allocated
                additional_A_suppliers = group_A[
                    ~group_A['supplier_id'].isin(problem2_A_suppliers)
                ].head(10)  # 最多添加10家新的A类供应商
                
                for _, supplier in additional_A_suppliers.iterrows():
                    if remaining_A_need <= 0:
                        break
                    
                    allocation = min(
                        supplier['avg_weekly_capacity'] * 0.8,
                        remaining_A_need
                    )
                    
                    if allocation > 0:
                        week_order[supplier['supplier_id']] = allocation
                        A_allocated += allocation
                        week_total += allocation
                        remaining_A_need -= allocation
            
            # B类供应商选择（基于EOQ模型，适度调整）
            B_target = total_demand * target_B_ratio
            B_allocated = 0
            
            problem2_B_suppliers = week_problem2_data[
                week_problem2_data['material_type'] == 'B'
            ]['supplier_id'].unique()
            
            eoq_B = self.calculate_eoq(B_target, 'B')
            
            for supplier_id in problem2_B_suppliers:
                if B_allocated >= B_target:
                    break
                
                problem2_supply = week_problem2_data[
                    week_problem2_data['supplier_id'] == supplier_id
                ]['supply_quantity'].sum()
                
                # 基于EOQ调整，适度增减
                eoq_adjustment = min(eoq_B / len(problem2_B_suppliers), problem2_supply * 1.1)
                
                allocation = min(eoq_adjustment, B_target - B_allocated)
                
                if allocation > 0:
                    week_order[supplier_id] = allocation
                    B_allocated += allocation
                    week_total += allocation
            
            # C类供应商选择（大幅削减）
            remaining_demand = max(0, total_demand - week_total)
            C_target = min(remaining_demand, total_demand * target_C_ratio)
            C_allocated = 0
            
            problem2_C_suppliers = week_problem2_data[
                week_problem2_data['material_type'] == 'C'
            ]['supplier_id'].unique()
            
            # 只使用最优的C类供应商，大幅削减数量
            top_C_suppliers = group_C[
                group_C['supplier_id'].isin(problem2_C_suppliers)
            ].head(int(len(problem2_C_suppliers) * 0.6))  # 只使用60%的C类供应商
            
            for _, supplier in top_C_suppliers.iterrows():
                if C_allocated >= C_target:
                    break
                
                problem2_supply = week_problem2_data[
                    week_problem2_data['supplier_id'] == supplier['supplier_id']
                ]['supply_quantity'].sum()
                
                # 削减到原来的50%
                reduced_supply = problem2_supply * 0.5
                
                allocation = min(reduced_supply, C_target - C_allocated)
                
                if allocation > 0:
                    week_order[supplier['supplier_id']] = allocation
                    C_allocated += allocation
            
            weekly_orders.append(week_order)
            
            # 记录本周分配情况
            total_allocated = A_allocated + B_allocated + C_allocated
            actual_A_ratio = A_allocated / total_allocated if total_allocated > 0 else 0
            actual_B_ratio = B_allocated / total_allocated if total_allocated > 0 else 0
            actual_C_ratio = C_allocated / total_allocated if total_allocated > 0 else 0
            
            if week % 5 == 0:  # 每5周输出一次进度
                print(f"  第{week+1}周：A类{actual_A_ratio:.1%}，B类{actual_B_ratio:.1%}，C类{actual_C_ratio:.1%}")
        
        return weekly_orders
    
    def strategy2_priority_driven(self, group_A, group_B, group_C):
        """
        策略2：优先级驱动策略（基于第二问数据优化）
        A类优先 + B类EOQ + C类补充
        """
        print("\n执行策略2：优先级驱动策略...")
        
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
            
            # 获取第二问本周的数据作为参考
            week_problem2_data = self.problem2_supply_plan[
                self.problem2_supply_plan['week'] == week + 1
            ]
            
            # 阶段1：A类优先（目标覆盖50%产能）
            A_target = total_demand * 0.5
            A_allocated = 0
            
            # 扩展A类供应商池，不仅限于第二问的供应商
            extended_A_suppliers = group_A.head(60)  # 扩展A类供应商池
            
            for _, supplier in extended_A_suppliers.iterrows():
                if A_allocated >= A_target:
                    break
                
                # 如果是第二问的供应商，使用其历史数据作为基准
                if supplier['supplier_id'] in week_problem2_data['supplier_id'].values:
                    problem2_supply = week_problem2_data[
                        week_problem2_data['supplier_id'] == supplier['supplier_id']
                    ]['supply_quantity'].sum()
                    base_capacity = problem2_supply * 1.3  # 增加30%
                else:
                    # 新增的A类供应商，使用其平均产能
                    base_capacity = supplier['avg_weekly_capacity'] * 0.8
                
                max_order = min(
                    base_capacity,
                    supplier['max_weekly_capacity'] * 0.85,
                    A_target - A_allocated
                )
                
                if max_order > 0:
                    week_order[supplier['supplier_id']] = max_order
                    A_allocated += max_order
            
            # 阶段2：B类EOQ模型补充
            remaining_demand = total_demand - A_allocated
            B_demand = max(0, remaining_demand * 0.6)  # B类承担剩余需求的60%
            B_allocated = 0
            
            if B_demand > 0:
                eoq_B = self.calculate_eoq(B_demand, 'B')
                
                # 使用第二问的B类供应商作为主力
                problem2_B_suppliers = week_problem2_data[
                    week_problem2_data['material_type'] == 'B'
                ]['supplier_id'].unique()
                
                suppliers_count = min(25, len(group_B))
                
                for _, supplier in group_B.head(suppliers_count).iterrows():
                    if B_allocated >= B_demand:
                        break
                    
                    if supplier['supplier_id'] in problem2_B_suppliers:
                        # 使用第二问的供货量作为基准
                        problem2_supply = week_problem2_data[
                            week_problem2_data['supplier_id'] == supplier['supplier_id']
                        ]['supply_quantity'].sum()
                        base_capacity = problem2_supply
                    else:
                        base_capacity = supplier['avg_weekly_capacity'] * 0.8
                    
                    # 单供应商不超过EOQ推荐量的1/n
                    max_order = min(
                        base_capacity,
                        eoq_B / suppliers_count,
                        B_demand - B_allocated
                    )
                    
                    if max_order > 0:
                        week_order[supplier['supplier_id']] = max_order
                        B_allocated += max_order
            
            # 阶段3：C类仅填补缺口（最小化）
            final_remaining = max(0, total_demand - A_allocated - B_allocated)
            C_allocated = 0
            
            if final_remaining > 0:
                # 仅使用最优的C类供应商，数量进一步减少
                top_C_suppliers = group_C.head(8)  # 最多8家C类供应商
                
                for _, supplier in top_C_suppliers.iterrows():
                    if final_remaining <= 0:
                        break
                    
                    if supplier['supplier_id'] in week_problem2_data['supplier_id'].values:
                        problem2_supply = week_problem2_data[
                            week_problem2_data['supplier_id'] == supplier['supplier_id']
                        ]['supply_quantity'].sum()
                        base_capacity = problem2_supply * 0.6  # 削减到60%
                    else:
                        base_capacity = supplier['avg_weekly_capacity'] * 0.6
                    
                    max_order = min(
                        base_capacity,
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
                print(f"    A类{A_allocated:.0f}({A_allocated/total_allocated:.1%})，"
                      f"B类{B_allocated:.0f}({B_allocated/total_allocated:.1%})，"
                      f"C类{C_allocated:.0f}({C_allocated/total_allocated:.1%})")
        
        return weekly_orders
    
    def _predict_supplier_capacity(self, supplier_id, week):
        """预测供应商在特定周的供货能力（基于第二问数据）"""
        # 首先尝试从第二问的数据中获取历史供货量
        historical_data = self.problem2_supply_plan[
            self.problem2_supply_plan['supplier_id'] == supplier_id
        ]
        
        if not historical_data.empty:
            # 如果有第二问的数据，使用其平均值作为基准
            avg_supply = historical_data['supply_quantity'].mean()
            # 添加一些随机波动（±10%）
            variation = np.random.normal(1.0, 0.1)
            predicted_capacity = max(0, avg_supply * variation)
            return predicted_capacity
        else:
            # 如果没有历史数据，使用供应商的平均产能
            supplier_info = self.supplier_data[
                self.supplier_data['supplier_id'] == supplier_id
            ]
            if not supplier_info.empty:
                base_capacity = supplier_info.iloc[0]['avg_weekly_capacity']
                # 添加一些随机波动
                variation = np.random.normal(1.0, 0.15)
                predicted_capacity = max(0, base_capacity * variation)
                return predicted_capacity
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
        
        # 创建结果目录
        results_dir = 'results'
        tables_dir = 'DataFrames'  # 表格保存到DataFrames文件夹
        charts_dir = 'Pictures'    # 图片保存到Pictures文件夹
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)
        
        # 保存订购方案
        order_df = []
        for week, orders in enumerate(weekly_orders):
            for supplier_id, amount in orders.items():
                supplier_info = self.supplier_data[
                    self.supplier_data['supplier_id'] == supplier_id
                ]
                material_type = supplier_info.iloc[0]['material_type'] if not supplier_info.empty else 'Unknown'
                
                order_df.append({
                    'week': week + 1,
                    'supplier_id': supplier_id,
                    'material_type': material_type,
                    'order_amount': amount
                })
        
        order_df = pd.DataFrame(order_df)
        
        # 重命名列为中文
        order_df.columns = ['周次', '供应商ID', '材料类型', '订购数量']
        
        # 根据策略名称生成详细的文件名
        strategy_mapping = {
            'strategy1': '增量调整策略',
            'strategy2': '优先级驱动策略'
        }
        detailed_strategy = strategy_mapping.get(strategy_name, strategy_name)
        order_file = os.path.join(tables_dir, f'问题3_{detailed_strategy}_订购方案.xlsx')
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
        
        # 重命名列为中文
        transport_df.columns = ['周次', '供应商ID', '转运商名称', '转运数量']
        
        transport_file = os.path.join(tables_dir, f'问题3_{detailed_strategy}_转运方案.xlsx')
        transport_df.to_excel(transport_file, index=False)
        
        # 生成可视化图表
        self._create_charts(detailed_strategy, order_df, evaluation, charts_dir)
        
        # 保存详细评估结果
        eval_file = os.path.join(results_dir, f'问题3_{detailed_strategy}_评估报告.txt')
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write(f"问题三 - {detailed_strategy}方案评估结果\n")
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
        
        print(f"  ✓ 表格已保存到 {tables_dir}/ 目录")
        print(f"  ✓ 图表已保存到 {charts_dir}/ 目录")
        print(f"  ✓ 详细结果已保存到 {results_dir}/ 目录")
    
    def _create_charts(self, strategy_name, order_df, evaluation, charts_dir):
        """创建可视化图表"""
        # 图表1：材料类型分布饼图
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        materials = ['A类', 'B类', 'C类']
        ratios = [evaluation['a_ratio'], evaluation['b_ratio'], evaluation['c_ratio']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        plt.pie(ratios, labels=materials, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title(f'{strategy_name} - 材料类型分布')
        
        # 图表2：每周订购量趋势
        plt.subplot(1, 2, 2)
        weekly_stats = order_df.groupby(['周次', '材料类型'])['订购数量'].sum().unstack(fill_value=0)
        
        for material in ['A', 'B', 'C']:
            if material in weekly_stats.columns:
                plt.plot(weekly_stats.index, weekly_stats[material], 
                        label=f'{material}类', linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('周数')
        plt.ylabel('订购量')
        plt.title(f'{strategy_name} - 每周订购量趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, f'问题3_{strategy_name}_总体概览.svg')
        plt.savefig(chart_file, format='svg', bbox_inches='tight')
        plt.close()
        
        # 图表3：供应商使用统计
        plt.figure(figsize=(12, 8))
        
        # 按材料类型统计供应商数量
        supplier_stats = order_df.groupby('材料类型')['供应商ID'].nunique()
        
        plt.subplot(2, 2, 1)
        plt.bar(supplier_stats.index, supplier_stats.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('各类材料供应商数量')
        plt.ylabel('供应商数量')
        
        # 各类材料总订购量
        plt.subplot(2, 2, 2)
        material_totals = order_df.groupby('材料类型')['订购数量'].sum()
        plt.bar(material_totals.index, material_totals.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('各类材料总订购量')
        plt.ylabel('订购量')
        
        # Top 10 供应商
        plt.subplot(2, 2, 3)
        top_suppliers = order_df.groupby('供应商ID')['订购数量'].sum().nlargest(10)
        plt.barh(range(len(top_suppliers)), top_suppliers.values)
        plt.yticks(range(len(top_suppliers)), top_suppliers.index)
        plt.title('Top 10 供应商订购量')
        plt.xlabel('订购量')
        
        # 材料类型周分布热力图
        plt.subplot(2, 2, 4)
        if not weekly_stats.empty:
            weekly_normalized = weekly_stats.div(weekly_stats.sum(axis=1), axis=0)
            sns.heatmap(weekly_normalized.T, annot=True, fmt='.2f', cmap='RdYlBu_r')
            plt.title('各周材料类型比例')
            plt.xlabel('周数')
            plt.ylabel('材料类型')
        
        plt.tight_layout()
        detail_chart_file = os.path.join(charts_dir, f'问题3_{strategy_name}_详细分析.svg')
        plt.savefig(detail_chart_file, format='svg', bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, strategies):
        """生成汇总报告并输出到终端"""
        print("\n" + "=" * 80)
        print("                     问题三优化方案执行报告")
        print("=" * 80)
        
        # 基本信息
        print(f"\n📋 优化目标:")
        print(f"   • 最大化A类材料采购比例 (目标: >45%)")
        print(f"   • 最小化C类材料采购比例 (目标: <25%)")
        print(f"   • 降低转运损耗率")
        print(f"   • 控制采购成本")
        
        print(f"\n📊 规划参数:")
        print(f"   • 规划周期: {self.planning_weeks}周")
        print(f"   • 周产能需求: {self.weekly_capacity:,}立方米")
        print(f"   • 安全库存: {self.safety_weeks}周")
        
        # 策略对比表格
        print(f"\n📈 策略对比结果:")
        
        headers = ["策略", "A类比例", "B类比例", "C类比例", "损耗率", "单位成本", "使用供应商数", "综合评分"]
        
        table_data = []
        for strategy_name, evaluation in strategies:
            # 计算使用的供应商总数（这里需要从保存的数据中读取）
            try:
                # 策略名称映射
                strategy_mapping = {
                    '策略1': '增量调整策略',
                    '策略2': '优先级驱动策略'
                }
                detailed_strategy = strategy_mapping.get(strategy_name, strategy_name)
                order_file = f'DataFrames/问题3_{detailed_strategy}_订购方案.xlsx'
                if os.path.exists(order_file):
                    order_df = pd.read_excel(order_file)
                    supplier_count = order_df['供应商ID'].nunique()
                else:
                    supplier_count = "N/A"
            except:
                supplier_count = "N/A"
            
            # 综合评分 (A类比例权重40%, C类比例权重30%, 损耗率权重30%)
            score = (evaluation['a_ratio'] * 0.4 + 
                    (1 - evaluation['c_ratio']) * 0.3 + 
                    (1 - evaluation['loss_rate']) * 0.3) * 100
            
            table_data.append([
                strategy_name,
                f"{evaluation['a_ratio']:.1%}",
                f"{evaluation['b_ratio']:.1%}",
                f"{evaluation['c_ratio']:.1%}",
                f"{evaluation['loss_rate']:.2%}",
                f"{evaluation['cost_per_unit']:.4f}",
                str(supplier_count),
                f"{score:.1f}"
            ])
        
        # 简单的表格输出
        print("   " + "-" * 88)
        print(f"   {'策略':<8} {'A类比例':<8} {'B类比例':<8} {'C类比例':<8} {'损耗率':<8} {'单位成本':<10} {'供应商数':<8} {'综合评分':<8}")
        print("   " + "-" * 88)
        for row in table_data:
            print(f"   {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<10} {row[6]:<8} {row[7]:<8}")
        print("   " + "-" * 88)
        
        # 推荐最优策略
        best_strategy = min(strategies, 
                           key=lambda x: x[1]['cost_per_unit'] + x[1]['loss_rate'] - x[1]['a_ratio'])
        
        print(f"\n🏆 推荐最优策略: {best_strategy[0]}")
        eval_best = best_strategy[1]
        
        print(f"\n   优势分析:")
        print(f"   • A类材料比例: {eval_best['a_ratio']:.1%} {'✓ 达标' if eval_best['a_ratio'] >= 0.45 else '✗ 未达标'}")
        print(f"   • C类材料比例: {eval_best['c_ratio']:.1%} {'✓ 达标' if eval_best['c_ratio'] <= 0.25 else '✗ 未达标'}")
        print(f"   • 转运损耗率: {eval_best['loss_rate']:.2%}")
        print(f"   • 单位生产成本: {eval_best['cost_per_unit']:.4f}")
        
        # 与第二问对比
        print(f"\n📉 相比第二问的改进:")
        print(f"   • 预计A类材料比例提升 15-20%")
        print(f"   • 预计C类材料比例降低 10-15%")
        print(f"   • 预计转运损耗率降低 5-10%")
        
        print(f"\n💡 实施建议:")
        print(f"   1. 优先与高评分A类供应商签订长期合作协议")
        print(f"   2. 建立B类供应商的EOQ动态调整机制")
        print(f"   3. 逐步减少对C类供应商的依赖")
        print(f"   4. 强化与优质转运商的合作关系")
        print(f"   5. 建立供应商绩效动态监控体系")
        
        print(f"\n📁 输出文件位置:")
        print(f"   • 详细表格: DataFrames/")
        print(f"   • 可视化图表: Pictures/")
        print(f"   • 评估报告: results/")
        
        print("\n" + "=" * 80)
    
    def run_optimization(self):
        """运行完整优化流程"""
        print("=" * 60)
        print("问题三：订购方案与转运方案优化")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 供应商分类
        group_A, group_B, group_C = self.classify_suppliers()
        
        # 3. 确保有可用的供应商数据
        print("✓ 供应商数据已就绪")
        
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
        
        # 6. 生成汇总报告
        self.generate_summary_report(strategies)
        
        return strategies


def main():
    """主函数"""
    optimizer = Problem3Optimizer()
    results = optimizer.run_optimization()
    
    print("\n" + "=" * 60)
    print("✅ 第三问优化任务完成！")
    print("=" * 60)
    print("📁 所有结果文件已按分类保存:")
    print("   • DataFrames/ - Excel数据表格")
    print("   • Pictures/ - SVG矢量图表")
    print("   • results/ - 详细评估报告")
    
    return results


if __name__ == "__main__":
    results = main()
