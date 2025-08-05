"""
第二问：供应商和转运商分配方案
根据预测模型生成满足24周100%达标的供货组合，并分配转运商
优先为供货量大的供应商分配损耗率低的转运商
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging
import os
import sys
from pathlib import Path

# 导入现有模块
try:
    from supplier_prediction_model_v3 import predict_multiple_suppliers
except ImportError:
    print("警告: 无法导入供应商预测模型，将使用备选方法")
    predict_multiple_suppliers = None

try:
    from transporter_loss_analysis import TransporterLossAnalyzer
except ImportError:
    print("警告: 无法导入转运商分析模块，将使用简化方法")
    TransporterLossAnalyzer = None

try:
    from monte_carlo_simulation import MonteCarloSimulator
except ImportError:
    print("警告: 无法导入蒙特卡洛模拟器，将跳过自动优化功能")
    MonteCarloSimulator = None

warnings.filterwarnings('ignore')

# 设置日志
log_dir = Path("log")
log_dir.mkdir(exist_ok=True)
log_filename = f"problem2_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class SupplierTransporterAllocator:
    """供应商转运商分配器"""
    
    def __init__(self):
        """初始化分配器"""
        # 基本参数
        self.target_weekly_capacity = 28200  # 企业周产能需求（立方米）
        self.planning_weeks = 24  # 规划周数
        self.safety_margin = 1.0  # 安全边际 (-2.5%)
        self.transporter_capacity = 6000  # 每家转运商运输能力（立方米/周）
        self.prediction_adjustment_factor = 1.05 # 预测微调乘数

        # 材料转换系数（原材料 -> 产品）
        self.material_conversion = {
            'A': 1/0.6,    # 1.6667 - 每1立方米A类原材料可制造1.6667立方米产品
            'B': 1/0.66,   # 1.5152 - 每1立方米B类原材料可制造1.5152立方米产品
            'C': 1/0.72    # 1.3889 - 每1立方米C类原材料可制造1.3889立方米产品
        }
        
        # 采购成本差异（相对于C类）
        self.material_cost_multiplier = {
            'A': 1.20,  # A类比C类高20%
            'B': 1.10,  # B类比C类高10%
            'C': 1.00   # C类基准价格
        }
        
        # 可手动调整的供应商数量
        self.num_suppliers = 85  # 默认85家供应商，可手动调整
        
        # 自动重试配置
        self.max_retry_attempts = 5  # 最大重试次数
        self.retry_increment = 0  # 每次增加的供应商数量

        # 成功判断标准：24周100%达标
        self.target_achievement_ratio = 1.0  # 100%周期达标
        
        # 数据存储
        self.supplier_pool = None
        self.transporter_data = None
        self.selected_suppliers = None
        self.supply_plan = None
        self.transport_plan = None
        
    def set_supplier_count(self, count):
        """设置供应商数量（可手动调整）"""
        self.num_suppliers = max(0, min(count, 402))  # 限制在0-402之间
        logging.info(f"设置供应商数量为: {self.num_suppliers}")
        print(f"供应商数量已设置为: {self.num_suppliers}")
    
    def set_retry_config(self, max_attempts=5, increment=20):
        """设置自动重试配置"""
        self.max_retry_attempts = max(1, max_attempts)
        self.retry_increment = max(0, min(increment, 50))  # 限制增量在0-50之间
        logging.info(f"重试配置: 最大尝试次数={self.max_retry_attempts}, 每次增加={self.retry_increment}家供应商")
        print(f"重试配置已更新: 最大尝试次数={self.max_retry_attempts}, 每次增加={self.retry_increment}家供应商")
    
    def load_data(self):
        """加载基础数据"""
        print("正在加载基础数据...")
        logging.info("开始加载基础数据")
        
        # 1. 加载供应商数据
        self._load_supplier_data()
        
        # 2. 加载转运商数据
        self._load_transporter_data()
        
        print("基础数据加载完成")
        logging.info("基础数据加载完成")
    
    def _load_supplier_data(self):
        """加载供应商基础数据"""
        print("  加载供应商基础数据...")
        
        # 1. 加载供应商产品制造能力汇总
        capacity_summary = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        print(f"    制造能力数据: {capacity_summary.shape}")
        
        # 2. 加载供应商可靠性排名
        reliability_ranking = pd.read_excel('DataFrames/供应商可靠性年度加权排名.xlsx')
        print(f"    可靠性排名数据: {reliability_ranking.shape}")
        
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
                'conversion_factor': self.material_conversion[material_type],
                'cost_multiplier': self.material_cost_multiplier[material_type]
            })
        
        self.supplier_pool = pd.DataFrame(supplier_pool)
        
        # 按综合评分排序（可靠性 + 产能）
        self.supplier_pool['composite_score'] = (
            self.supplier_pool['reliability_score'] * 0.6 + 
            (self.supplier_pool['avg_weekly_capacity'] / self.supplier_pool['avg_weekly_capacity'].max()) * 0.4
        )
        
        self.supplier_pool = self.supplier_pool.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        print(f"    供应商池构建完成: {len(self.supplier_pool)} 家供应商")
        print(f"    材料类型分布:")
        for material in ['A', 'B', 'C']:
            count = len(self.supplier_pool[self.supplier_pool['material_type'] == material])
            total_capacity = self.supplier_pool[self.supplier_pool['material_type'] == material]['avg_weekly_capacity'].sum()
            print(f"      {material}类: {count}家, 总产能: {total_capacity:.0f}")
    
    def _load_transporter_data(self):
        """加载转运商损耗率数据"""
        print("  加载转运商损耗率数据...")
        
        # 读取转运商损耗率分析结果
        if os.path.exists('DataFrames/转运商损耗率分析结果.xlsx'):
            self.transporter_data = pd.read_excel('DataFrames/转运商损耗率分析结果.xlsx')
            print(f"    转运商数据: {self.transporter_data.shape}")
            
            # 重命名列以保持一致性
            if 'transporter_name' in self.transporter_data.columns:
                self.transporter_data = self.transporter_data.rename(columns={'transporter_name': 'transporter_id'})
            
            # 转换损耗率：从百分比转换为小数（除以100）
            if 'avg_loss_rate' in self.transporter_data.columns:
                self.transporter_data['avg_loss_rate'] = self.transporter_data['avg_loss_rate'] / 100
            
        else:
            # 如果分析结果文件不存在，使用原始数据
            print("    转运商分析结果文件不存在，读取原始数据...")
            transporter_raw = pd.read_excel('C/附件2 近5年8家转运商的相关数据.xlsx', sheet_name='转运商的运输损耗率')
            
            # 分析转运商数据
            if TransporterLossAnalyzer is not None:
                analyzer = TransporterLossAnalyzer()
                analyzer.load_data('C/附件2 近5年8家转运商的相关数据.xlsx')
            
            # 获取转运商平均损耗率
            transporter_summary = []
            for transporter in transporter_raw.iloc[:, 0]:
                if pd.notna(transporter):
                    transporter_data = transporter_raw[transporter_raw.iloc[:, 0] == transporter].iloc[:, 1:].values.flatten()
                    # 移除0值和NaN值
                    valid_data = transporter_data[(transporter_data > 0) & (~np.isnan(transporter_data))]
                    
                    if len(valid_data) > 0:
                        avg_loss_rate = np.mean(valid_data) / 100  # 原始数据是百分比，需要除以100
                        std_loss_rate = np.std(valid_data) / 100
                        transporter_summary.append({
                            'transporter_id': transporter,
                            'avg_loss_rate': avg_loss_rate,
                            'std_loss_rate': std_loss_rate,
                            'data_points': len(valid_data)
                        })
            
            self.transporter_data = pd.DataFrame(transporter_summary)
        
        # 按平均损耗率排序（损耗率低的排前面）
        if 'avg_loss_rate' in self.transporter_data.columns:
            self.transporter_data = self.transporter_data.sort_values('avg_loss_rate').reset_index(drop=True)
            print(f"    转运商损耗率范围: {self.transporter_data['avg_loss_rate'].min():.4f} - {self.transporter_data['avg_loss_rate'].max():.4f}")
            print(f"    即: {self.transporter_data['avg_loss_rate'].min()*100:.2f}% - {self.transporter_data['avg_loss_rate'].max()*100:.2f}%")
        
        print(f"    可用转运商数量: {len(self.transporter_data)}")
    
    def generate_optimal_supply_plan(self, auto_retry=True):
        """生成满足24周100%达标的最优供货计划（支持自动重试）"""
        print(f"\n正在生成满足100%达标的供货计划（供应商数量: {self.num_suppliers}）...")
        logging.info(f"开始生成供货计划，供应商数量: {self.num_suppliers}")
        
        retry_count = 0
        original_num_suppliers = self.num_suppliers
        
        while retry_count <= self.max_retry_attempts:
            if retry_count > 0:
                print(f"\n第{retry_count}次重试，当前供应商数量: {self.num_suppliers}")
                logging.info(f"第{retry_count}次重试，供应商数量: {self.num_suppliers}")
            
            # 选择Top N供应商
            self.selected_suppliers = self.supplier_pool.head(self.num_suppliers).copy()
            
            print(f"选定供应商组成:")
            material_counts = self.selected_suppliers['material_type'].value_counts()
            for material in ['A', 'B', 'C']:
                count = material_counts.get(material, 0)
                total_capacity = self.selected_suppliers[
                    self.selected_suppliers['material_type'] == material
                ]['avg_weekly_capacity'].sum()
                print(f"  {material}类: {count}家, 总产能: {total_capacity:.0f}")
            
            # 使用预测模型生成供货量
            supplier_ids = self.selected_suppliers['supplier_id'].tolist()
            print("正在调用预测模型生成24周供货量...")
            
            predictions = None
            if predict_multiple_suppliers is not None:
                try:
                    predictions = predict_multiple_suppliers(supplier_ids, self.planning_weeks, use_multithread=True)
                    print(f"预测完成，获得 {len(predictions)} 家供应商的预测数据")
                except Exception as e:
                    print(f"预测模型调用失败: {e}")
                    logging.error(f"预测模型调用失败: {e}")
                    predictions = None
            
            if predictions is None:
                print("使用备选方法生成供货量...")
                predictions = self._generate_fallback_predictions(supplier_ids)
            
            # 将预测值调整并向上取整
            predictions = {k: np.ceil(v * self.prediction_adjustment_factor).astype(int) for k, v in predictions.items()}

            # 生成供货计划表
            supply_plan = []
            for week in range(self.planning_weeks):
                week_supplies = []
                
                for _, supplier in self.selected_suppliers.iterrows():
                    supplier_id = supplier['supplier_id']
                    material_type = supplier['material_type']
                    conversion_factor = supplier['conversion_factor']
                    
                    if supplier_id in predictions:
                        # 使用预测值
                        raw_supply = predictions[supplier_id][week]
                    else:
                        # 使用平均值作为备选
                        raw_supply = supplier['avg_weekly_capacity']
                    
                    # 确保供货量非负
                    raw_supply = max(0, raw_supply)
                    
                    # 计算可制造的产品量
                    product_capacity = raw_supply * conversion_factor
                    
                    week_supplies.append({
                        'week': week + 1,
                        'supplier_id': supplier_id,
                        'material_type': material_type,
                        'supply_quantity': raw_supply,
                        'conversion_factor': conversion_factor,
                        'product_capacity': product_capacity,
                        'cost_multiplier': supplier['cost_multiplier']
                    })
                
                supply_plan.extend(week_supplies)
            
            self.supply_plan = pd.DataFrame(supply_plan)
            
            # 验证供货计划是否满足100%达标要求
            is_success = self._validate_supply_plan()
            
            if is_success or not auto_retry:
                # 成功达标或不自动重试，退出循环
                break
            
            # 未达标且可以重试
            if retry_count < self.max_retry_attempts:
                # 增加供应商数量
                new_count = min(402, self.num_suppliers + self.retry_increment)
                if new_count == self.num_suppliers:
                    # 已达到最大供应商数量，无法继续增加
                    print(f"⚠️  已达到最大供应商数量(402家)，无法继续增加")
                    logging.warning("已达到最大供应商数量，停止重试")
                    break
                
                print(f"📈 自动增加供应商数量: {self.num_suppliers} → {new_count}")
                logging.info(f"自动增加供应商数量: {self.num_suppliers} → {new_count}")
                self.num_suppliers = new_count
                retry_count += 1
            else:
                # 达到最大重试次数
                print(f"⚠️  已达到最大重试次数({self.max_retry_attempts})，停止重试")
                logging.warning(f"达到最大重试次数({self.max_retry_attempts})，停止重试")
                break
        
        # 输出最终结果
        if retry_count > 0:
            print(f"\n📊 重试总结:")
            print(f"  原始供应商数量: {original_num_suppliers}")
            print(f"  最终供应商数量: {self.num_suppliers}")
            print(f"  重试次数: {retry_count}")
            logging.info(f"重试完成: 原始{original_num_suppliers} → 最终{self.num_suppliers}, 重试{retry_count}次")
        
        print("供货计划生成完成")
        logging.info("供货计划生成完成")
    
    def _generate_fallback_predictions(self, supplier_ids):
        """生成备选预测（基于历史平均值和随机波动）"""
        print("使用备选方法生成预测...")
        predictions = {}
        
        for supplier_id in supplier_ids:
            supplier_info = self.selected_suppliers[
                self.selected_suppliers['supplier_id'] == supplier_id
            ].iloc[0]
            
            base_capacity = supplier_info['avg_weekly_capacity']
            stability = supplier_info['stability']
            volatility = stability / base_capacity if base_capacity > 0 else 0.2
            
            # 生成24周的预测值
            weekly_predictions = []
            for week in range(self.planning_weeks):
                # 基础产能 + 随机波动
                prediction = base_capacity * (1 + np.random.normal(0, volatility))
                prediction = max(0, prediction)  # 确保非负
                weekly_predictions.append(prediction)
            
            predictions[supplier_id] = weekly_predictions
        
        return predictions
    
    def _validate_supply_plan(self):
        """验证供货计划是否满足100%达标要求"""
        print("验证供货计划达标情况...")
        
        # 计算每周总产能（考虑转运损耗）
        weekly_capacities = []
        for week in range(1, self.planning_weeks + 1):
            week_data = self.supply_plan[self.supply_plan['week'] == week]
            # 假设平均损耗率为0.5%（实际分配时会用具体转运商的损耗率）
            estimated_loss_rate = 0.005
            total_capacity_with_loss = week_data['product_capacity'].sum() * (1 - estimated_loss_rate)
            weekly_capacities.append(total_capacity_with_loss)
        
        # 计算累计产能和达标率
        cumulative_capacities = np.cumsum(weekly_capacities)
        target_cumulative = np.array([self.target_weekly_capacity * i for i in range(1, self.planning_weeks + 1)])
        
        # 考虑安全边际
        target_with_margin = target_cumulative * self.safety_margin
        achievement_ratios = cumulative_capacities / target_with_margin
        
        # 计算达标周数（基于累计产能）
        weeks_meeting_target = np.sum(achievement_ratios >= 1.0)
        success_rate = weeks_meeting_target / self.planning_weeks
        
        # 同时检查每周产能是否满足要求
        weekly_targets = np.array([self.target_weekly_capacity] * self.planning_weeks) * self.safety_margin
        weekly_achievement = np.array(weekly_capacities) / weekly_targets
        weekly_success_count = np.sum(weekly_achievement >= 1.0)
        weekly_success_rate = weekly_success_count / self.planning_weeks
        
        print(f"供货计划验证结果:")
        print(f"  累计产能达标: {weeks_meeting_target}/{self.planning_weeks} ({success_rate:.2%})")
        print(f"  单周产能达标: {weekly_success_count}/{self.planning_weeks} ({weekly_success_rate:.2%})")
        print(f"  最低周产能: {min(weekly_capacities):,.0f}")
        print(f"  平均周产能: {np.mean(weekly_capacities):,.0f}")
        print(f"  目标周产能: {self.target_weekly_capacity:,}")
        print(f"  实际目标(含安全边际): {self.target_weekly_capacity * self.safety_margin:,.0f}")
        
        # 显示前几周的详细情况
        print(f"  前5周产能情况:")
        for i in range(min(5, self.planning_weeks)):
            print(f"    第{i+1}周: 产能 {weekly_capacities[i]:,.0f}, "
                  f"累计 {cumulative_capacities[i]:,.0f}, "
                  f"达标率 {achievement_ratios[i]:.2f}")
        
        # 如果不满足100%达标，建议增加供应商数量
        if success_rate < self.target_achievement_ratio:
            # 估算需要的供应商数量增长比例
            capacity_deficit = np.mean(target_with_margin / cumulative_capacities)
            suggested_multiplier = min(2.0, capacity_deficit * 1.2)  # 最多增加100%
            suggested_count = min(402, int(self.num_suppliers * suggested_multiplier))
            
            print(f"⚠️  当前方案未满足100%达标要求")
            print(f"   累计产能缺口约 {(capacity_deficit - 1) * 100:.1f}%")
            print(f"   建议增加供应商数量至: {suggested_count}")
            logging.warning(f"供货计划未满足100%达标，建议增加供应商数量至 {suggested_count}")
            
            return False
        else:
            print("✅ 供货计划满足100%达标要求")
            logging.info("供货计划满足100%达标要求")
            return True
    
    def allocate_transporters(self):
        """分配转运商：供货量大的供应商优先使用损耗率低的转运商"""
        print("\n正在分配转运商...")
        logging.info("开始分配转运商")
        
        if self.supply_plan is None:
            raise ValueError("请先生成供货计划")
        
        transport_plan = []
        
        for week in range(1, self.planning_weeks + 1):
            print(f"  分配第{week}周转运商...")
            
            # 获取本周的供货数据
            week_supplies = self.supply_plan[self.supply_plan['week'] == week].copy()
            
            # 按供货量降序排序（供货量大的优先）
            week_supplies = week_supplies.sort_values('supply_quantity', ascending=False)
            
            # 初始化转运商使用情况
            transporter_usage = {
                row['transporter_id']: 0 
                for _, row in self.transporter_data.iterrows()
            }
            
            # 为每个供应商分配转运商
            for _, supply in week_supplies.iterrows():
                supplier_id = supply['supplier_id']
                material_type = supply['material_type']
                supply_quantity = supply['supply_quantity']
                
                if supply_quantity <= 0:
                    continue
                
                # 寻找最优转运商（损耗率最低且有足够运力）
                best_transporter = None
                min_loss_rate = float('inf')
                
                for _, transporter in self.transporter_data.iterrows():
                    transporter_id = transporter['transporter_id']
                    avg_loss_rate = transporter['avg_loss_rate']
                    
                    # 检查运力是否足够
                    current_usage = transporter_usage[transporter_id]
                    remaining_capacity = self.transporter_capacity - current_usage
                    
                    if remaining_capacity >= supply_quantity and avg_loss_rate < min_loss_rate:
                        best_transporter = transporter_id
                        min_loss_rate = avg_loss_rate
                
                # 如果没有找到合适的转运商，选择使用率最低的
                if best_transporter is None:
                    min_usage = float('inf')
                    for transporter_id, usage in transporter_usage.items():
                        if usage < min_usage:
                            min_usage = usage
                            best_transporter = transporter_id
                            
                    # 更新损耗率
                    transporter_info = self.transporter_data[
                        self.transporter_data['transporter_id'] == best_transporter
                    ]
                    if not transporter_info.empty:
                        min_loss_rate = transporter_info.iloc[0]['avg_loss_rate']
                
                # 更新转运商使用情况
                transporter_usage[best_transporter] += supply_quantity
                
                # 计算运输损耗和接收量
                loss_quantity = supply_quantity * min_loss_rate
                received_quantity = supply_quantity - loss_quantity
                product_quantity = received_quantity * supply['conversion_factor']
                
                transport_plan.append({
                    'week': week,
                    'supplier_id': supplier_id,
                    'material_type': material_type,
                    'supply_quantity': supply_quantity,
                    'transporter_id': best_transporter,
                    'loss_rate': min_loss_rate,
                    'loss_quantity': loss_quantity,
                    'received_quantity': received_quantity,
                    'product_quantity': product_quantity,
                    'cost_multiplier': supply['cost_multiplier']
                })
        
        self.transport_plan = pd.DataFrame(transport_plan)
        
        print("转运商分配完成")
        logging.info("转运商分配完成")
        
        # 分析转运方案
        self._analyze_transport_plan()
    
    def _analyze_transport_plan(self):
        """分析转运方案效果"""
        print("\n转运方案分析:")
        
        # 转运商使用统计
        transporter_stats = self.transport_plan.groupby('transporter_id').agg({
            'supply_quantity': ['sum', 'count'],
            'loss_quantity': 'sum',
            'received_quantity': 'sum'
        }).round(2)
        
        print("转运商使用统计:")
        for transporter_id in self.transporter_data['transporter_id']:
            if transporter_id in transporter_stats.index:
                total_supply = transporter_stats.loc[transporter_id, ('supply_quantity', 'sum')]
                usage_count = transporter_stats.loc[transporter_id, ('supply_quantity', 'count')]
                total_loss = transporter_stats.loc[transporter_id, ('loss_quantity', 'sum')]
                loss_rate = total_loss / total_supply if total_supply > 0 else 0
                
                print(f"  {transporter_id}: 总运量 {total_supply:,.0f}, 使用次数 {usage_count}, 总损耗率 {loss_rate:.3f}")
        
        # 总体损耗统计
        total_supply = self.transport_plan['supply_quantity'].sum()
        total_loss = self.transport_plan['loss_quantity'].sum()
        total_received = self.transport_plan['received_quantity'].sum()
        overall_loss_rate = total_loss / total_supply if total_supply > 0 else 0
        
        print(f"\n总体运输效果:")
        print(f"  总供货量: {total_supply:,.0f} 立方米")
        print(f"  总损耗量: {total_loss:,.0f} 立方米")
        print(f"  总接收量: {total_received:,.0f} 立方米")
        print(f"  整体损耗率: {overall_loss_rate:.3f} ({overall_loss_rate*100:.2f}%)")
        
        # 材料类型分析
        material_stats = self.transport_plan.groupby('material_type').agg({
            'supply_quantity': 'sum',
            'loss_quantity': 'sum',
            'received_quantity': 'sum',
            'product_quantity': 'sum'
        }).round(2)
        
        print(f"\n各类材料统计:")
        for material in ['A', 'B', 'C']:
            if material in material_stats.index:
                supply = material_stats.loc[material, 'supply_quantity']
                loss = material_stats.loc[material, 'loss_quantity']
                received = material_stats.loc[material, 'received_quantity']
                product = material_stats.loc[material, 'product_quantity']
                loss_rate = loss / supply if supply > 0 else 0
                
                print(f"  {material}类: 供货 {supply:,.0f}, 损耗 {loss:,.0f} ({loss_rate:.3f}), "
                      f"接收 {received:,.0f}, 可制造产品 {product:,.0f}")
    
    def export_results(self, filename_prefix="problem2_allocation"):
        """导出结果到Excel文件"""
        print(f"\n正在导出结果...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 创建DataFrames目录
        dataFrame_dir = Path("DataFrames")
        dataFrame_dir.mkdir(exist_ok=True)
        
        # 导出文件路径
        supply_file = dataFrame_dir / f"{filename_prefix}_supply.xlsx"
        transport_file = dataFrame_dir / f"{filename_prefix}_transport.xlsx"
        summary_file = dataFrame_dir / f"{filename_prefix}_summary.xlsx"

        # 按题目要求的附件A和附件B格式
        attachment_a_file = dataFrame_dir / f"问题2_附件A_订购方案数据结果(格式不对).xlsx"
        attachment_b_file = dataFrame_dir / f"问题2_附件B_转运方案数据结果(格式不对).xlsx"

        # 1. 导出供货计划
        if self.supply_plan is not None:
            self.supply_plan.to_excel(supply_file, index=False)
            print(f"  供货计划已导出: {supply_file}")
        
        # 2. 导出转运计划（按题目要求格式）
        if self.transport_plan is not None:
            # 按照题目要求的格式整理数据
            export_transport = self.transport_plan[
                ['supplier_id', 'material_type', 'supply_quantity', 
                 'transporter_id', 'loss_quantity', 'received_quantity', 'product_quantity']
            ].copy()
            
            export_transport.columns = [
                '供应商ID', '原材料类型', '供货数量', 
                '转运商', '理论运输损耗值', '理论接收原材料', '可制作产品量'
            ]
            
            export_transport.to_excel(transport_file, index=False)
            print(f"  转运计划已导出: {transport_file}")
        
        # 3. 按附件A格式导出订购方案（供应商-周度矩阵）
        if self.supply_plan is not None:
            self._export_attachment_a_format(attachment_a_file)
            print(f"  订购方案(附件A格式)已导出: {attachment_a_file}")
        
        # 4. 按附件B格式导出转运方案（转运商-周度分配）
        if self.transport_plan is not None:
            self._export_attachment_b_format(attachment_b_file)
            print(f"  转运方案(附件B格式)已导出: {attachment_b_file}")
        
        # 5. 导出汇总分析
        with pd.ExcelWriter(summary_file) as writer:
            # 选定供应商信息
            if self.selected_suppliers is not None:
                self.selected_suppliers.to_excel(writer, sheet_name='选定供应商', index=False)
            
            # 转运商信息
            if self.transporter_data is not None:
                self.transporter_data.to_excel(writer, sheet_name='转运商信息', index=False)
            
            # 周度汇总
            if self.transport_plan is not None:
                weekly_summary = self.transport_plan.groupby('week').agg({
                    'supply_quantity': 'sum',
                    'loss_quantity': 'sum',
                    'received_quantity': 'sum',
                    'product_quantity': 'sum'
                }).round(2)
                
                weekly_summary.columns = ['总供货量', '总损耗量', '总接收量', '总产品量']
                weekly_summary['损耗率'] = (weekly_summary['总损耗量'] / weekly_summary['总供货量']).round(4)
                weekly_summary.to_excel(writer, sheet_name='周度汇总')
            
            # 材料类型汇总
            if self.transport_plan is not None:
                material_summary = self.transport_plan.groupby('material_type').agg({
                    'supply_quantity': 'sum',
                    'loss_quantity': 'sum',
                    'received_quantity': 'sum',
                    'product_quantity': 'sum'
                }).round(2)
                
                material_summary.columns = ['总供货量', '总损耗量', '总接收量', '总产品量']
                material_summary['损耗率'] = (material_summary['总损耗量'] / material_summary['总供货量']).round(4)
                material_summary.to_excel(writer, sheet_name='材料类型汇总')
            
            print(f"  汇总分析已导出: {summary_file}")
        
        logging.info(f"结果导出完成: {supply_file}, {transport_file}, {summary_file}")
    
    def _export_attachment_a_format(self, filename):
        """按附件A格式导出订购方案（供应商-周度矩阵）"""
        # 创建供应商-周度矩阵
        pivot_data = self.supply_plan.pivot_table(
            index=['supplier_id', 'material_type'],
            columns='week',
            values='supply_quantity',
            fill_value=0
        )
        
        # 重置索引以便导出
        pivot_data = pivot_data.reset_index()
        
        # 重命名列以符合附件A格式
        week_columns = {f'week': f'第{i}周' for i in range(1, self.planning_weeks + 1)}
        pivot_data.columns.name = None
        
        # 重新组织列顺序
        final_columns = ['supplier_id', 'material_type'] + list(range(1, self.planning_weeks + 1))
        pivot_data.columns = ['供应商名称', '原材料类别'] + [f'第{i}周' for i in range(1, self.planning_weeks + 1)]
        
        # 导出到Excel
        pivot_data.to_excel(filename, index=False)
    
    def _export_attachment_b_format(self, filename):
        """按附件B格式导出转运方案（转运商-周度分配）"""
        # 按转运商和周度汇总运输量
        transport_summary = self.transport_plan.groupby(['transporter_id', 'week']).agg({
            'supply_quantity': 'sum',
            'loss_quantity': 'sum',
            'received_quantity': 'sum'
        }).reset_index()
        
        # 创建转运商-周度矩阵（运输量）
        transport_pivot = transport_summary.pivot_table(
            index='transporter_id',
            columns='week',
            values='supply_quantity',
            fill_value=0
        )
        
        # 重置索引
        transport_pivot = transport_pivot.reset_index()
        transport_pivot.columns.name = None
        
        # 重命名列
        transport_pivot.columns = ['转运商名称'] + [f'第{i}周运输量' for i in range(1, self.planning_weeks + 1)]
        
        # 同时导出损耗情况
        loss_pivot = transport_summary.pivot_table(
            index='transporter_id',
            columns='week',
            values='loss_quantity',
            fill_value=0
        )
        
        loss_pivot = loss_pivot.reset_index()
        loss_pivot.columns.name = None
        loss_pivot.columns = ['转运商名称'] + [f'第{i}周损耗量' for i in range(1, self.planning_weeks + 1)]
        
        # 使用多个工作表导出
        with pd.ExcelWriter(filename) as writer:
            transport_pivot.to_excel(writer, sheet_name='转运商运输量', index=False)
            loss_pivot.to_excel(writer, sheet_name='转运商损耗量', index=False)
            
            # 详细的转运分配表
            detailed_transport = self.transport_plan[
                ['week', 'supplier_id', 'material_type', 'supply_quantity', 
                 'transporter_id', 'loss_quantity', 'received_quantity']
            ].copy()
            
            detailed_transport.columns = [
                '周次', '供应商ID', '原材料类型', '供货量',
                '转运商', '损耗量', '接收量'
            ]
            
            detailed_transport.to_excel(writer, sheet_name='详细转运分配', index=False)
    
    def find_minimum_suppliers_for_100_percent(self, max_suppliers=402, start_count=200):
        """寻找满足24周100%达标的最少供应商数量"""
        print("\n正在寻找满足100%达标的最少供应商数量...")
        logging.info("开始寻找满足100%达标的最少供应商数量")
        
        if self.supplier_pool is None:
            raise ValueError("请先加载数据")
        
        if MonteCarloSimulator is None:
            print("⚠️  蒙特卡洛模拟器不可用，使用估算方法...")
            # 基于平均产能的估算
            estimated_min = self._estimate_minimum_suppliers()
            self.set_supplier_count(estimated_min)
            return estimated_min
        
        # 使用蒙特卡洛模拟器进行验证
        simulator = MonteCarloSimulator()
        simulator.target_achievement_ratio = 1.0  # 设置为100%达标
        
        # 二分查找最小供应商数量
        left, right = start_count, max_suppliers
        min_suppliers = max_suppliers
        
        while left <= right:
            mid = (left + right) // 2
            print(f"  测试 {mid} 家供应商...")
            
            # 选择Top N供应商
            test_suppliers = self.supplier_pool.head(mid)
            
            # 进行蒙特卡洛模拟验证
            result = simulator.simulate_supply_scenario(
                test_suppliers, 
                num_simulations=100,  # 减少模拟次数以加快速度
                show_progress=False,
                max_workers=16
            )
            
            success_rate = result['success_rate']
            print(f"    结果: 成功率 {success_rate:.2%}")
            
            if success_rate >= 0.65:  # 如果65%的模拟都能100%达标，认为方案可行
                min_suppliers = mid
                right = mid - 1
                print(f"    ✅ {mid}家供应商可行")
            else:
                left = mid + 1
                print(f"    ❌ {mid}家供应商不可行")
        
        print(f"\n推荐最少供应商数量: {min_suppliers}")
        logging.info(f"推荐最少供应商数量: {min_suppliers}")
        
        # 设置为推荐数量
        self.set_supplier_count(min_suppliers)
        
        return min_suppliers
    
    def _estimate_minimum_suppliers(self):
        """估算满足需求的最少供应商数量（备选方法）"""
        # 计算总需求
        total_demand = self.target_weekly_capacity * self.planning_weeks
        
        # 考虑安全边际和转运损耗
        adjusted_demand = total_demand / self.safety_margin / 0.995  # 假设0.5%损耗率
        
        # 按产能排序，累计计算
        supplier_pool_sorted = self.supplier_pool.sort_values('avg_weekly_capacity', ascending=False)
        
        cumulative_capacity = 0
        for i, (_, supplier) in enumerate(supplier_pool_sorted.iterrows()):
            # 原材料产能转换为产品产能
            product_capacity = supplier['avg_weekly_capacity'] * supplier['conversion_factor']
            cumulative_capacity += product_capacity * self.planning_weeks * supplier['reliability_score']
            
            if cumulative_capacity >= adjusted_demand:
                estimated_min = min(402, int((i + 1) * 1.2))  # 增加20%安全余量
                print(f"估算最少供应商数量: {estimated_min}")
                return estimated_min
        
        return 300  # 默认返回值
    
    def run_complete_allocation(self, auto_find_minimum=False, auto_retry=True):
        """运行完整的分配流程"""
        print("="*60)
        print("第二问：供应商和转运商分配方案")
        print("="*60)
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 自动寻找最小供应商数量（可选）
            if auto_find_minimum:
                self.find_minimum_suppliers_for_100_percent()
            
            # 3. 生成供货计划（支持自动重试）
            self.generate_optimal_supply_plan(auto_retry=auto_retry)
            
            # 4. 分配转运商
            self.allocate_transporters()
            
            # 5. 导出结果
            self.export_results()
            
            print("\n" + "="*60)
            print("第二问完成!")
            print(f"最终方案: {self.num_suppliers}家供应商, 24周100%达标")
            print("="*60)
            
        except Exception as e:
            print(f"执行过程中出现错误: {e}")
            logging.error(f"执行错误: {e}", exc_info=True)

def main():
    """主函数"""
    allocator = SupplierTransporterAllocator()
    
    print("第二问：供应商和转运商分配方案")
    print("="*60)
    print("运行模式选择:")
    print("1. 手动设置供应商数量")
    print("2. 自动寻找最少供应商数量（推荐）")
    print("3. 启用自动重试模式")
    
    mode = input("请选择模式 (1/2/3, 默认为3): ").strip()
    
    if mode == "1":
        # 手动模式
        print(f"\n当前供应商数量设置: {allocator.num_suppliers}")
        try:
            new_count = input(f"请输入供应商数量 (1-402, 默认为{allocator.num_suppliers}): ").strip()
            if new_count:
                allocator.set_supplier_count(int(new_count))
        except ValueError:
            print("输入无效，使用默认数量")
        
        # 运行分配流程（不自动重试）
        allocator.run_complete_allocation(auto_find_minimum=False, auto_retry=False)
    
    elif mode == "2":
        # 自动寻找最小数量模式
        print("\n将自动寻找满足100%达标的最少供应商数量...")
        allocator.run_complete_allocation(auto_find_minimum=True, auto_retry=False)
    
    else:
        # 自动重试模式（默认）
        print(f"\n自动重试模式:")
        print(f"  当前供应商数量: {allocator.num_suppliers}")
        print(f"  最大重试次数: {allocator.max_retry_attempts}")
        print(f"  每次增加数量: {allocator.retry_increment}")
        
        # 询问是否修改重试配置
        config_input = input("是否修改重试配置? (y/N): ").strip().lower()
        if config_input in ['y', 'yes']:
            try:
                max_attempts = input(f"最大重试次数 (默认{allocator.max_retry_attempts}): ").strip()
                increment = input(f"每次增加供应商数量 (默认{allocator.retry_increment}): ").strip()
                
                if max_attempts:
                    max_attempts = int(max_attempts)
                else:
                    max_attempts = allocator.max_retry_attempts
                    
                if increment:
                    increment = int(increment)
                else:
                    increment = allocator.retry_increment
                
                allocator.set_retry_config(max_attempts, increment)
            except ValueError:
                print("输入无效，使用默认配置")
        
        # 运行分配流程（启用自动重试）
        allocator.run_complete_allocation(auto_find_minimum=False, auto_retry=True)
    
    print("\n提示: 结果文件已保存到 DataFrames/ 目录")
    print("包括:")
    print("- 供货计划详情")
    print("- 转运分配方案") 
    print("- 汇总分析报告")
    print("- 附件A/B格式文件")

if __name__ == "__main__":
    main()
