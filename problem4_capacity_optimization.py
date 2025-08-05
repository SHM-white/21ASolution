"""
问题四：产能提升潜力分析与最优方案
根据现有原材料供应商和转运商的实际情况，确定企业每周产能可以提高多少，
并给出未来24周的订购和转运方案。

分析思路：
1. 基于现有402家供应商的最大供应能力分析
2. 考虑8家转运商的运输限制
3. 在确保供应安全的前提下，最大化产能利用
4. 输出格式与第二问、第三问保持一致
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging
import os
from pathlib import Path

# 导入预测模型
from supplier_prediction_model_v3 import get_trained_timeseries_model, predict_multiple_suppliers

warnings.filterwarnings('ignore')

class Problem4CapacityOptimizer:
    """问题四产能优化器"""
    
    def __init__(self):
        # 基本参数
        self.current_weekly_capacity = 28200  # 当前周产能（立方米）
        self.planning_weeks = 24             # 规划周数
        self.safety_weeks = 2                # 安全库存周数
        
        # 原材料转换比例
        self.material_conversion = {
            'A': 0.6,   # A类原材料需求量
            'B': 0.66,  # B类原材料需求量  
            'C': 0.72   # C类原材料需求量
        }
        
        # 转运商运输能力
        self.transporter_capacity = 6000  # 立方米/周
        
        # 数据存储
        self.supplier_data = None
        self.transporter_data = None
        self.optimal_capacity = None
        self.supply_plan = []
        self.transport_plan = []
        
        # 预测模型可用性
        self.prediction_model_available = False
        
    def load_data(self):
        """加载基础数据"""
        print("加载基础数据...")
        
        # 1. 加载供应商制造能力数据
        capacity_df = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        
        # 2. 加载供应商可靠性排名
        reliability_df = pd.read_excel('DataFrames/供应商可靠性年度加权排名.xlsx')
        
        # 3. 加载转运商数据
        transporter_df = pd.read_excel('DataFrames/转运商损耗率分析结果.xlsx')
        
        # 4. 检查预测模型可用性
        print("检查预测模型...")
        try:
            # 简单测试预测模型是否可用
            test_result = predict_multiple_suppliers(['S001'], 1, use_multithread=False)
            if test_result and 'S001' in test_result:
                self.prediction_model_available = True
                print("✓ 预测模型可用")
            else:
                self.prediction_model_available = False
                print("⚠ 预测模型测试失败")
        except Exception as e:
            self.prediction_model_available = False
            print(f"⚠ 预测模型不可用: {e}")
        
        # 5. 构建供应商数据池
        supplier_pool = []
        
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
            
            is_top50 = not reliability_info.empty
            reliability_score = reliability_info['加权可靠性得分'].iloc[0] if is_top50 else 0.5
            
            # 计算产品制造能力（考虑转换比例）
            conversion_factor = 1 / self.material_conversion[material_type]
            avg_product_capacity = avg_capacity * conversion_factor
            max_product_capacity = max_capacity * conversion_factor
            
            supplier_pool.append({
                'supplier_id': supplier_id,
                'material_type': material_type,
                'avg_weekly_capacity': avg_capacity,
                'max_weekly_capacity': max_capacity,
                'avg_product_capacity': avg_product_capacity,
                'max_product_capacity': max_product_capacity,
                'stability': stability,
                'reliability_score': reliability_score,
                'is_top50': is_top50,
                'conversion_factor': conversion_factor
            })
        
        self.supplier_data = pd.DataFrame(supplier_pool)
        
        # 5. 处理转运商数据
        self.transporter_data = transporter_df[['transporter_name', 'avg_loss_rate', 
                                               'stability_score', 'comprehensive_score']].copy()
        
        print(f"✓ 加载完成：{len(self.supplier_data)} 家供应商，{len(self.transporter_data)} 家转运商")
        
        # 显示材料类型分布
        for material in ['A', 'B', 'C']:
            material_suppliers = self.supplier_data[self.supplier_data['material_type'] == material]
            total_avg_capacity = material_suppliers['avg_weekly_capacity'].sum()
            total_max_capacity = material_suppliers['max_weekly_capacity'].sum()
            top50_count = material_suppliers['is_top50'].sum()
            
            print(f"  {material}类: {len(material_suppliers)}家供应商, "
                  f"平均产能: {total_avg_capacity:,.0f}, "
                  f"最大产能: {total_max_capacity:,.0f}, "
                  f"Top50: {top50_count}家")
    
    def analyze_capacity_potential(self):
        """分析产能提升潜力（基于预测模型）"""
        print("\n基于预测模型分析产能提升潜力...")
        
        # 1. 使用预测模型预测各供应商未来供货能力
        material_capacity_limits = {}
        
        for material in ['A', 'B', 'C']:
            material_suppliers = self.supplier_data[self.supplier_data['material_type'] == material]
            supplier_ids = material_suppliers['supplier_id'].tolist()
            
            print(f"  分析{material}类材料供应商({len(supplier_ids)}家)...")
            
            # 使用预测模型预测未来4周的供货量
            if self.prediction_model_available:
                try:
                    # 使用直接的批量预测函数
                    prediction_results = predict_multiple_suppliers(
                        supplier_ids, 
                        prediction_weeks=4, 
                        use_multithread=True
                    )
                    
                    # 计算各供应商的平均预测产能
                    total_predicted_capacity = 0
                    valid_predictions = 0
                    
                    for supplier_id in supplier_ids:
                        if supplier_id in prediction_results:
                            predictions = prediction_results[supplier_id]
                            # 取预测值的90%分位数，避免极端值影响
                            avg_prediction = np.percentile(predictions, 90)

                            # 获取历史数据作为参考
                            supplier_info = material_suppliers[material_suppliers['supplier_id'] == supplier_id]
                            if not supplier_info.empty:
                                max_capacity = supplier_info['max_weekly_capacity'].iloc[0]
                                avg_capacity = supplier_info['avg_weekly_capacity'].iloc[0]
                                
                                # 使用预测值和历史最大值的加权平均，更均衡的配比
                                # 预测值60%，历史最大值的80%权重40%
                                optimistic_prediction = 0.6 * avg_prediction + 0.4 * (max_capacity * 0.8)
                                
                                # 限制在合理范围内：不超过历史最大值的130%，不低于历史平均值的50%
                                optimistic_prediction = min(optimistic_prediction, max_capacity * 1.3)
                                optimistic_prediction = max(optimistic_prediction, avg_capacity * 0.5)
                            else:
                                optimistic_prediction = avg_prediction
                            
                            total_predicted_capacity += max(optimistic_prediction, 0)
                            valid_predictions += 1
                    
                    calculation_method = f"预测模型+历史数据加权({valid_predictions}/{len(supplier_ids)}家供应商)"
                    
                    if valid_predictions == 0:
                        # 如果预测失败，回退到历史最大值的85%
                        total_predicted_capacity = material_suppliers['max_weekly_capacity'].sum() * 0.85
                        calculation_method = "历史最大值85%（预测模型失败回退方案）"
                        
                except Exception as e:
                    print(f"    预测模型调用失败: {e}")
                    # 回退到历史最大值的85%
                    total_predicted_capacity = material_suppliers['max_weekly_capacity'].sum() * 0.85
                    calculation_method = "历史最大值85%（预测模型异常回退方案）"
            else:
                # 如果预测模型不可用，使用历史最大值的85%
                total_predicted_capacity = material_suppliers['max_weekly_capacity'].sum() * 0.85
                calculation_method = "历史最大值85%（无预测模型）"
            
            # 转换为产品制造能力
            product_capacity = total_predicted_capacity / self.material_conversion[material]
            material_capacity_limits[material] = product_capacity
            
            print(f"    {material}类材料可支持最大产能: {product_capacity:,.0f} 立方米/周 ({calculation_method})")
        
        # 2. 供应商限制的最大产能（受最小材料类型限制）
        supplier_limited_capacity = min(material_capacity_limits.values())
        print(f"\n供应商能力限制的最大产能: {supplier_limited_capacity:,.0f} 立方米/周")
        
        # 3. 转运商限制分析
        total_transport_capacity = len(self.transporter_data) * self.transporter_capacity
        # 考虑平均损耗率（约1.5%）
        effective_transport_capacity = total_transport_capacity * 0.985
        
        print(f"转运商能力限制分析:")
        print(f"  - 8家转运商×6000立方米/周 = {total_transport_capacity:,.0f} 立方米/周")
        print(f"  - 考虑1.5%损耗后有效运力: {effective_transport_capacity:,.0f} 立方米/周")
        
        # 转运商限制的产品制造能力受原材料转换比例限制
        # 按照当前产能的原材料配比计算：A类30%、B类33%、C类37%
        weighted_conversion = 0.3 * self.material_conversion['A'] + 0.33 * self.material_conversion['B'] + 0.37 * self.material_conversion['C']
        transport_limited_capacity = effective_transport_capacity / weighted_conversion
        
        print(f"  - 转换为产品制造能力: {transport_limited_capacity:,.0f} 立方米/周")
        print(f"  - 加权平均原材料消耗率: {weighted_conversion:.3f}")
        
        # 4. 实际可达到的最大产能
        self.optimal_capacity = min(supplier_limited_capacity, transport_limited_capacity)
        capacity_increase = self.optimal_capacity - self.current_weekly_capacity
        increase_percentage = capacity_increase / self.current_weekly_capacity * 100
        
        print(f"\n=== 产能提升分析结果 ===")
        print(f"分析方法说明: 基于时间序列预测模型的可持续供应能力")
        print(f"当前产能: {self.current_weekly_capacity:,.0f} 立方米/周")
        print(f"最大可达产能: {self.optimal_capacity:,.0f} 立方米/周")
        print(f"可提升产能: {capacity_increase:,.0f} 立方米/周")
        print(f"提升幅度: {increase_percentage:.1f}%")
        
        if capacity_increase > 0:
            print(f"✓ 有产能提升空间")
        else:
            print(f"⚠ 当前产能已接近极限，受转运商能力限制")
            
        return self.optimal_capacity
    
    def generate_optimal_supply_plan(self):
        """生成最优供货计划，考虑转运能力约束和预测模型"""
        print(f"\n生成基于{self.optimal_capacity:,.0f}立方米/周产能的供货计划...")
        
        # 每周最大转运能力
        max_weekly_transport = 8 * self.transporter_capacity  # 8个转运商 × 6000立方米
        print(f"每周最大转运能力: {max_weekly_transport:,.0f} 立方米")
        
        # 计算目标产能所需的原材料总量
        # 按照当前产能的原材料配比计算：A类30%、B类33%、C类37%
        weighted_conversion = 0.3 * self.material_conversion['A'] + 0.33 * self.material_conversion['B'] + 0.37 * self.material_conversion['C']
        required_materials = self.optimal_capacity * weighted_conversion
        
        print(f"目标产能({self.optimal_capacity:,.0f}立方米/周)所需原材料: {required_materials:,.0f} 立方米/周")
        print(f"加权平均原材料消耗率: {weighted_conversion:.3f}")
        
        # 检查是否超过转运能力
        if required_materials > max_weekly_transport:  # 留5%余量
            print(f"⚠ 警告：所需原材料({required_materials:,.0f})超过转运能力({max_weekly_transport:,.0f})，调整目标产能")
            adjusted_capacity = (max_weekly_transport) / weighted_conversion
            print(f"调整后目标产能: {adjusted_capacity:,.0f} 立方米/周")
        else:
            adjusted_capacity = self.optimal_capacity
            print(f"✓ 转运能力充足，维持目标产能: {adjusted_capacity:,.0f} 立方米/周")
        
        # 计算每周原材料需求
        weekly_demands = {}
        total_material_demand = 0
        for material, consumption in self.material_conversion.items():
            weekly_demands[material] = adjusted_capacity * consumption
            total_material_demand += weekly_demands[material]
            print(f"  {material}类原材料需求: {weekly_demands[material]:,.0f} 立方米/周")
        
        print(f"  总原材料需求: {total_material_demand:,.0f} 立方米/周")
        
        # 检查总需求是否超过转运能力，如果超过则按比例调整
        if total_material_demand > max_weekly_transport:
            print(f"⚠ 警告：总原材料需求({total_material_demand:,.0f})超过转运能力({max_weekly_transport:,.0f})")
            adjustment_factor = (max_weekly_transport) / total_material_demand
            print(f"按比例调整系数: {adjustment_factor:.3f}")
            
            for material in weekly_demands:
                weekly_demands[material] *= adjustment_factor
                print(f"  调整后{material}类需求: {weekly_demands[material]:,.0f} 立方米/周")
        
        # 获取各供应商的预测供货能力
        supplier_predicted_capacity = {}
        if self.prediction_model_available:
            print("  获取供应商预测供货能力...")
            try:
                all_supplier_ids = self.supplier_data['supplier_id'].tolist()
                # 使用直接的批量预测函数
                prediction_results = predict_multiple_suppliers(
                    all_supplier_ids, 
                    prediction_weeks=4,  # 预测未来4周取平均
                    use_multithread=True
                )
                
                for supplier_id, predictions in prediction_results.items():
                    # 取预测平均值，但限制在合理范围内
                    avg_prediction = np.mean(predictions)
                    supplier_info = self.supplier_data[self.supplier_data['supplier_id'] == supplier_id]
                    if not supplier_info.empty:
                        max_capacity = supplier_info['max_weekly_capacity'].iloc[0]
                        # 限制预测值在历史最大值的120%以内
                        avg_prediction = min(avg_prediction, max_capacity * 1.2)
                        avg_prediction = max(avg_prediction, max_capacity * 0.3)  # 至少30%
                    
                    supplier_predicted_capacity[supplier_id] = avg_prediction
                    
                print(f"    ✓ 获取{len(supplier_predicted_capacity)}家供应商的预测供货能力")
            except Exception as e:
                print(f"    ⚠ 预测供货能力失败: {e}")
                supplier_predicted_capacity = {}
        
        # 选择供应商策略：优先选择高可靠性和高预测产能的供应商
        for week in range(1, self.planning_weeks + 1):
            week_supplies = []
            week_total_supply = 0
            
            for material in ['A', 'B', 'C']:
                material_demand = weekly_demands[material]
                
                # 按可靠性和预测产能排序选择供应商
                material_suppliers = self.supplier_data[
                    self.supplier_data['material_type'] == material
                ].copy()
                
                # 添加预测产能信息
                if supplier_predicted_capacity:
                    material_suppliers['predicted_capacity'] = material_suppliers['supplier_id'].map(
                        lambda x: supplier_predicted_capacity.get(x, material_suppliers[
                            material_suppliers['supplier_id'] == x]['max_weekly_capacity'].iloc[0] * 0.7 
                            if not material_suppliers[material_suppliers['supplier_id'] == x].empty else 100
                        )
                    )
                    # 按预测产能和可靠性排序
                    material_suppliers = material_suppliers.sort_values(
                        ['is_top50', 'reliability_score', 'predicted_capacity'], 
                        ascending=[False, False, False]
                    )
                else:
                    # 如果没有预测数据，按原方式排序
                    material_suppliers = material_suppliers.sort_values(
                        ['is_top50', 'reliability_score', 'max_weekly_capacity'], 
                        ascending=[False, False, False]
                    )
                
                allocated_demand = 0
                supplier_count = 0
                
                for _, supplier in material_suppliers.iterrows():
                    if allocated_demand >= material_demand:
                        break
                    
                    # 检查是否会超过周转运能力
                    if week_total_supply >= max_weekly_transport:
                        break
                    
                    # 确定供应商的供货能力
                    if supplier_predicted_capacity and supplier['supplier_id'] in supplier_predicted_capacity:
                        # 使用预测产能，但不超过70%（保持稳定性）
                        base_capacity = supplier_predicted_capacity[supplier['supplier_id']] * 0.7
                    else:
                        # 使用历史最大值的70%
                        base_capacity = supplier['max_weekly_capacity'] * 0.7
                    
                    # 限制单个供应商的供货量
                    supply_capacity = min(
                        base_capacity,
                        # 移除转运商单次运力限制，允许供应商向多个转运商供货
                        material_demand - allocated_demand,  # 不超过需求量
                        max_weekly_transport - week_total_supply  # 不超过周转运余量
                    )
                    
                    if supply_capacity > 10:  # 最低订货量10立方米
                        week_supplies.append({
                            'week': week,
                            'supplier_id': supplier['supplier_id'],
                            'material_type': material,
                            'supply_quantity': supply_capacity,
                            'conversion_factor': supplier['conversion_factor'],
                            'product_capacity': supply_capacity * supplier['conversion_factor'],
                            'reliability_score': supplier['reliability_score'],
                            'is_predicted': supplier['supplier_id'] in supplier_predicted_capacity
                        })
                        
                        allocated_demand += supply_capacity
                        week_total_supply += supply_capacity
                        supplier_count += 1
                
                if week == 1:  # 只在第一周显示详细信息
                    # 修正：只统计当前材料类型的供应商
                    if supplier_count > 0:
                        predicted_count = sum(1 for s in week_supplies[-supplier_count:] if s.get('is_predicted', False))
                    else:
                        predicted_count = 0
                    print(f"    第{week}周{material}类: 使用{supplier_count}家供应商(预测模型指导: {predicted_count}家), 分配{allocated_demand:,.0f}立方米")
            
            if week == 1:
                print(f"    第{week}周总供货量: {week_total_supply:,.0f} 立方米")
            
            self.supply_plan.extend(week_supplies)
        
        self.supply_plan = pd.DataFrame(self.supply_plan)
        total_weekly_avg = len(self.supply_plan) / 24 if len(self.supply_plan) > 0 else 0
        
        # 统计预测模型的使用情况
        predicted_records = self.supply_plan[self.supply_plan.get('is_predicted', False)].shape[0] if 'is_predicted' in self.supply_plan.columns else 0
        print(f"✓ 供货计划生成完成，共{len(self.supply_plan)}条记录，平均每周{total_weekly_avg:.1f}条")
        print(f"✓ 其中{predicted_records}条记录基于预测模型指导")
        
        # 清理不必要的列
        if 'is_predicted' in self.supply_plan.columns:
            self.supply_plan = self.supply_plan.drop('is_predicted', axis=1)
        
    def allocate_transporters(self):
        """分配转运商"""
        print("\n分配转运商...")
        
        # 按综合评分排序转运商（损耗率低、稳定性高优先）
        sorted_transporters = self.transporter_data.sort_values('comprehensive_score', ascending=False)
        
        for week in range(1, self.planning_weeks + 1):
            week_supplies = self.supply_plan[self.supply_plan['week'] == week].copy()
            week_supplies = week_supplies.sort_values('supply_quantity', ascending=False)
            
            # 转运商使用情况跟踪
            transporter_usage = {
                row['transporter_name']: 0 
                for _, row in self.transporter_data.iterrows()
            }
            
            for _, supply in week_supplies.iterrows():
                supplier_id = supply['supplier_id']
                material_type = supply['material_type']
                supply_quantity = supply['supply_quantity']
                
                # 优先寻找能够完整运输的转运商
                best_transporter = None
                min_loss_rate = float('inf')
                
                # 第一轮：寻找有足够剩余运力的转运商
                for _, transporter in sorted_transporters.iterrows():
                    transporter_name = transporter['transporter_name']
                    loss_rate = transporter['avg_loss_rate']
                    remaining_capacity = self.transporter_capacity - transporter_usage[transporter_name]
                    
                    # 如果该转运商能够完整运输
                    if remaining_capacity >= supply_quantity:
                        if loss_rate < min_loss_rate:
                            best_transporter = transporter_name
                            min_loss_rate = loss_rate
                
                # 如果找到了能完整运输的转运商，直接分配
                if best_transporter is not None:
                    transporter_usage[best_transporter] += supply_quantity
                    loss_quantity = supply_quantity * min_loss_rate / 100
                    received_quantity = supply_quantity - loss_quantity
                    
                    self.transport_plan.append({
                        'week': week,
                        'supplier_id': supplier_id,
                        'material_type': material_type,
                        'transporter_name': best_transporter,
                        'supply_quantity': supply_quantity,
                        'loss_rate': min_loss_rate,
                        'loss_quantity': loss_quantity,
                        'received_quantity': received_quantity
                    })
                else:
                    # 没有单个转运商能完整运输，需要分拆到多个转运商
                    remaining_quantity = supply_quantity
                    
                    while remaining_quantity > 0:
                        # 寻找最优转运商
                        best_transporter = None
                        max_available_capacity = 0
                        min_loss_rate = float('inf')
                        
                        for _, transporter in sorted_transporters.iterrows():
                            transporter_name = transporter['transporter_name']
                            loss_rate = transporter['avg_loss_rate']
                            
                            # 计算该转运商的剩余运力
                            remaining_capacity = self.transporter_capacity - transporter_usage[transporter_name]
                            
                            if remaining_capacity > 0:
                                # 优先选择损耗率低且有足够运力的转运商
                                if loss_rate < min_loss_rate or (loss_rate == min_loss_rate and remaining_capacity > max_available_capacity):
                                    best_transporter = transporter_name
                                    min_loss_rate = loss_rate
                                    max_available_capacity = remaining_capacity
                        
                        # 如果没有找到可用转运商，选择使用率最低的
                        if best_transporter is None:
                            best_transporter = min(transporter_usage, key=transporter_usage.get)
                            min_loss_rate = self.transporter_data[
                                self.transporter_data['transporter_name'] == best_transporter
                            ]['avg_loss_rate'].iloc[0]
                            max_available_capacity = self.transporter_capacity - transporter_usage[best_transporter]
                        
                        # 计算本次分配的数量（不超过转运商剩余运力）
                        allocated_quantity = min(remaining_quantity, max_available_capacity, self.transporter_capacity)
                        
                        if allocated_quantity <= 0:
                            # 所有转运商都已达到满负荷，无法继续分配
                            print(f"⚠ 警告：第{week}周转运商运力不足，剩余{remaining_quantity:.0f}立方米无法运输")
                            break
                        
                        # 更新使用情况
                        transporter_usage[best_transporter] += allocated_quantity
                        remaining_quantity -= allocated_quantity
                        
                        # 计算损耗和接收量
                        loss_quantity = allocated_quantity * min_loss_rate / 100
                        received_quantity = allocated_quantity - loss_quantity
                        
                        self.transport_plan.append({
                            'week': week,
                            'supplier_id': supplier_id,
                            'material_type': material_type,
                            'transporter_name': best_transporter,
                            'supply_quantity': allocated_quantity,
                            'loss_rate': min_loss_rate,
                            'loss_quantity': loss_quantity,
                            'received_quantity': received_quantity
                        })
        
        self.transport_plan = pd.DataFrame(self.transport_plan)
        print(f"✓ 转运方案生成完成，共{len(self.transport_plan)}条记录")
        
        # 分析转运效果
        self._analyze_transport_plan()
    
    def _analyze_transport_plan(self):
        """分析转运方案效果"""
        print("\n转运方案分析:")
        
        # 总体统计
        total_supply = self.transport_plan['supply_quantity'].sum()
        total_loss = self.transport_plan['loss_quantity'].sum()
        total_received = self.transport_plan['received_quantity'].sum()
        overall_loss_rate = total_loss / total_supply * 100
        
        print(f"  总供货量: {total_supply:,.0f} 立方米")
        print(f"  总损耗量: {total_loss:,.0f} 立方米")
        print(f"  总接收量: {total_received:,.0f} 立方米")
        print(f"  平均损耗率: {overall_loss_rate:.2f}%")
        
        # 转运商使用统计
        transporter_stats = self.transport_plan.groupby('transporter_name').agg({
            'supply_quantity': ['sum', 'count'],
            'loss_quantity': 'sum'
        }).round(2)
        
        print(f"\n转运商使用情况:")
        for transporter_name in self.transporter_data['transporter_name']:
            if transporter_name in transporter_stats.index:
                total_transport = transporter_stats.loc[transporter_name, ('supply_quantity', 'sum')]
                usage_count = transporter_stats.loc[transporter_name, ('supply_quantity', 'count')]
                total_loss = transporter_stats.loc[transporter_name, ('loss_quantity', 'sum')]
                loss_rate = total_loss / total_transport * 100 if total_transport > 0 else 0
                
                print(f"  {transporter_name}: 运输量 {total_transport:,.0f}, "
                      f"使用次数 {usage_count}, 损耗率 {loss_rate:.2f}%")
    
    def export_results(self):
        """导出结果到Excel文件"""
        print("\n导出结果...")
        
        # 确保DataFrames目录存在
        os.makedirs('DataFrames', exist_ok=True)
        
        # 1. 导出订购方案（标准格式）
        supply_result = self.supply_plan[['week', 'supplier_id', 'material_type', 'supply_quantity']].copy()
        supply_result.columns = ['周次', '供应商ID', '材料类型', '订购数量']
        supply_result.to_excel('DataFrames/问题4_订购方案.xlsx', index=False)
        
        # 2. 导出转运方案（标准格式）
        transport_result = self.transport_plan[['week', 'supplier_id', 'transporter_name', 'supply_quantity']].copy()
        transport_result.columns = ['周次', '供应商ID', '转运商名称', '转运数量']
        transport_result.to_excel('DataFrames/问题4_转运方案.xlsx', index=False)
        
        # 3. 导出详细分析报告
        summary_data = {
            '产能分析': [
                f'当前产能: {self.current_weekly_capacity:,.0f} 立方米/周',
                f'最优产能: {self.optimal_capacity:,.0f} 立方米/周',
                f'提升幅度: {(self.optimal_capacity/self.current_weekly_capacity-1)*100:.1f}%'
            ],
            '供应商使用': [
                f'总供应商数: {self.supply_plan["supplier_id"].nunique()}家',
                f'A类供应商: {len(self.supply_plan[self.supply_plan["material_type"]=="A"]["supplier_id"].unique())}家',
                f'B类供应商: {len(self.supply_plan[self.supply_plan["material_type"]=="B"]["supplier_id"].unique())}家',
                f'C类供应商: {len(self.supply_plan[self.supply_plan["material_type"]=="C"]["supplier_id"].unique())}家'
            ],
            '转运效果': [
                f'总运输量: {self.transport_plan["supply_quantity"].sum():,.0f} 立方米',
                f'总损耗量: {self.transport_plan["loss_quantity"].sum():,.0f} 立方米',
                f'平均损耗率: {self.transport_plan["loss_quantity"].sum()/self.transport_plan["supply_quantity"].sum()*100:.2f}%'
            ]
        }
        
        with pd.ExcelWriter('DataFrames/问题4_详细分析报告.xlsx') as writer:
            for sheet_name, data in summary_data.items():
                df = pd.DataFrame({'指标': range(1, len(data)+1), '内容': data})
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✓ 结果已导出到DataFrames/问题4_订购方案.xlsx")
        print(f"✓ 结果已导出到DataFrames/问题4_转运方案.xlsx")
        print(f"✓ 详细报告已导出到DataFrames/问题4_详细分析报告.xlsx")
    
    def run_optimization(self):
        """运行完整优化流程"""
        print("=" * 60)
        print("问题四：产能提升潜力分析与最优方案")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 分析产能提升潜力
        self.analyze_capacity_potential()
        
        # 3. 生成最优供货计划
        self.generate_optimal_supply_plan()
        
        # 4. 分配转运商
        self.allocate_transporters()
        
        # 5. 导出结果
        self.export_results()
        
        print("\n" + "=" * 60)
        print("问题四优化完成")
        print("=" * 60)
        
        return self.optimal_capacity

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log/problem4_optimization.log'),
            logging.StreamHandler()
        ]
    )
    
    optimizer = Problem4CapacityOptimizer()
    optimal_capacity = optimizer.run_optimization()
    
    print(f"\n最终结果：企业最优产能为 {optimal_capacity:,.0f} 立方米/周")
