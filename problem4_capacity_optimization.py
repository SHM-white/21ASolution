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
        
    def load_data(self):
        """加载基础数据"""
        print("加载基础数据...")
        
        # 1. 加载供应商制造能力数据
        capacity_df = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        
        # 2. 加载供应商可靠性排名
        reliability_df = pd.read_excel('DataFrames/供应商可靠性年度加权排名.xlsx')
        
        # 3. 加载转运商数据
        transporter_df = pd.read_excel('DataFrames/转运商损耗率分析结果.xlsx')
        
        # 4. 加载供应商90%分位数数据（用于可持续供应能力计算）
        try:
            percentile_df = pd.read_excel('DataFrames/供应商统计数据离散系数_重处理.xlsx')
            print(f"✓ 加载90%分位数数据：{len(percentile_df)}条记录")
        except FileNotFoundError:
            print("⚠ 警告：未找到重处理的离散系数文件，将使用最大产能的80%作为备选方案")
            percentile_df = None
        
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
        
        # 存储90%分位数数据供后续使用
        self.percentile_data = percentile_df
        
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
        """分析产能提升潜力"""
        print("\n分析产能提升潜力...")
        
        # 1. 分析各类材料的供应能力限制
        material_capacity_limits = {}
        
        for material in ['A', 'B', 'C']:
            material_suppliers = self.supplier_data[self.supplier_data['material_type'] == material]
            
            # 计算可持续供应能力：优先使用90%分位数，否则使用最大产能的80%
            if self.percentile_data is not None:
                # 使用90%分位数作为可持续供应能力
                material_percentile_data = self.percentile_data[
                    self.percentile_data['材料分类'] == material
                ]
                
                if not material_percentile_data.empty:
                    # 按供应商数量加权计算90%分位数总产能
                    sustainable_capacity = material_percentile_data['90%分位数'].sum()
                    calculation_method = "90%分位数"
                else:
                    # 如果没有找到对应材料的90%分位数数据，回退到80%方法
                    sustainable_capacity = material_suppliers['max_weekly_capacity'].sum() * 0.8
                    calculation_method = "最大产能的80%（回退方案）"
            else:
                # 使用最大产能的80%作为可持续供应能力（考虑稳定性）
                sustainable_capacity = material_suppliers['max_weekly_capacity'].sum() * 0.8
                calculation_method = "最大产能的80%（备选方案）"
            
            # 转换为产品制造能力
            product_capacity = sustainable_capacity / self.material_conversion[material]
            material_capacity_limits[material] = product_capacity
            
            print(f"  {material}类材料可支持最大产能: {product_capacity:,.0f} 立方米/周 ({calculation_method})")
        
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
        calculation_note = "（基于90%分位数可持续供应能力）" if self.percentile_data is not None else "（基于最大产能80%）"
        print(f"分析方法说明: 供应商可持续产能{calculation_note}")
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
        """生成最优供货计划，考虑转运能力约束"""
        print(f"\n生成基于{self.optimal_capacity:,.0f}立方米/周产能的供货计划...")
        
        # 每周最大转运能力
        max_weekly_transport = 8 * self.transporter_capacity  # 8个转运商 × 6000立方米
        print(f"每周最大转运能力: {max_weekly_transport:,.0f} 立方米")
        
        # 如果目标产能超过转运能力，需要调整
        if self.optimal_capacity > max_weekly_transport:
            print(f"⚠ 警告：目标产能({self.optimal_capacity:,.0f})超过转运能力({max_weekly_transport:,.0f})，调整目标产能")
            adjusted_capacity = max_weekly_transport * 0.95  # 留5%余量
            print(f"调整后目标产能: {adjusted_capacity:,.0f} 立方米/周")
        else:
            adjusted_capacity = self.optimal_capacity
        
        # 计算每周原材料需求
        weekly_demands = {}
        for material, consumption in self.material_conversion.items():
            weekly_demands[material] = adjusted_capacity * consumption
            print(f"  {material}类原材料需求: {weekly_demands[material]:,.0f} 立方米/周")
        
        # 选择供应商策略：优先选择高可靠性和高产能的供应商
        for week in range(1, self.planning_weeks + 1):
            week_supplies = []
            week_total_supply = 0
            
            for material in ['A', 'B', 'C']:
                material_demand = weekly_demands[material]
                
                # 按可靠性和产能排序选择供应商
                material_suppliers = self.supplier_data[
                    self.supplier_data['material_type'] == material
                ].copy()
                
                # 优先选择Top50供应商，然后按产能排序
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
                    if week_total_supply >= max_weekly_transport * 0.95:
                        break
                    
                    # 限制单个供应商的供货量，避免超过转运商单次运力
                    supply_capacity = min(
                        supplier['max_weekly_capacity'] * 0.7,
                        self.transporter_capacity,  # 不超过单个转运商能力
                        material_demand - allocated_demand,  # 不超过需求量
                        max_weekly_transport * 0.95 - week_total_supply  # 不超过周转运余量
                    )
                    
                    if supply_capacity > 10:  # 最低订货量10立方米
                        week_supplies.append({
                            'week': week,
                            'supplier_id': supplier['supplier_id'],
                            'material_type': material,
                            'supply_quantity': supply_capacity,
                            'conversion_factor': supplier['conversion_factor'],
                            'product_capacity': supply_capacity * supplier['conversion_factor'],
                            'reliability_score': supplier['reliability_score']
                        })
                        
                        allocated_demand += supply_capacity
                        week_total_supply += supply_capacity
                        supplier_count += 1
                
                if week == 1:  # 只在第一周显示详细信息
                    print(f"    第{week}周{material}类: 使用{supplier_count}家供应商, 分配{allocated_demand:,.0f}立方米")
            
            if week == 1:
                print(f"    第{week}周总供货量: {week_total_supply:,.0f} 立方米")
            
            self.supply_plan.extend(week_supplies)
        
        self.supply_plan = pd.DataFrame(self.supply_plan)
        total_weekly_avg = len(self.supply_plan) / 24 if len(self.supply_plan) > 0 else 0
        print(f"✓ 供货计划生成完成，共{len(self.supply_plan)}条记录，平均每周{total_weekly_avg:.1f}条")
        
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
                
                # 如果供货量超过单个转运商的运力，需要分拆到多个转运商
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
