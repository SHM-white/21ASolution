"""
问题三结果分析工具
分析并展示供应商组合和供货量按材料类型的分类
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False    # 正常显示负号

class Problem3ResultAnalyzer:
    """问题三结果分析器"""
    
    def __init__(self):
        self.supplier_data = None
        self.strategy1_orders = None
        self.strategy2_orders = None
        self.strategy1_transport = None
        self.strategy2_transport = None
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 加载供应商基础数据
        self.supplier_data = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
        
        # 加载策略结果
        self.strategy1_orders = pd.read_excel('results/problem3_strategy1_orders.xlsx')
        self.strategy2_orders = pd.read_excel('results/problem3_strategy2_orders.xlsx')
        self.strategy1_transport = pd.read_excel('results/problem3_strategy1_transport.xlsx')
        self.strategy2_transport = pd.read_excel('results/problem3_strategy2_transport.xlsx')
        
        print("✓ 数据加载完成")
    
    def analyze_supplier_composition(self, strategy_name, orders_df):
        """分析供应商组合构成"""
        print(f"\n=== {strategy_name} 供应商组合分析 ===")
        
        # 合并供应商信息
        analysis_df = orders_df.merge(
            self.supplier_data[['供应商ID', '材料分类', '平均周制造能力', '最大周制造能力']],
            left_on='supplier_id',
            right_on='供应商ID',
            how='left'
        )
        
        # 按材料类型分组统计
        material_stats = {}
        
        for material in ['A', 'B', 'C']:
            material_df = analysis_df[analysis_df['材料分类'] == material]
            
            if len(material_df) > 0:
                stats = {
                    'supplier_count': material_df['supplier_id'].nunique(),
                    'total_orders': material_df['order_amount'].sum(),
                    'avg_weekly_order': material_df.groupby('week')['order_amount'].sum().mean(),
                    'max_weekly_order': material_df.groupby('week')['order_amount'].sum().max(),
                    'min_weekly_order': material_df.groupby('week')['order_amount'].sum().min(),
                    'suppliers': material_df['supplier_id'].unique().tolist(),
                    'weekly_totals': material_df.groupby('week')['order_amount'].sum().tolist()
                }
            else:
                stats = {
                    'supplier_count': 0,
                    'total_orders': 0,
                    'avg_weekly_order': 0,
                    'max_weekly_order': 0,
                    'min_weekly_order': 0,
                    'suppliers': [],
                    'weekly_totals': [0] * 24
                }
            
            material_stats[material] = stats
        
        # 输出统计结果
        total_orders = sum([stats['total_orders'] for stats in material_stats.values()])
        
        print(f"\n总体统计:")
        print(f"  总订购量: {total_orders:,.0f} 立方米")
        print(f"  供应商总数: {sum([stats['supplier_count'] for stats in material_stats.values()])} 家")
        
        print(f"\n按材料类型分类:")
        for material, stats in material_stats.items():
            ratio = stats['total_orders'] / total_orders * 100 if total_orders > 0 else 0
            print(f"  {material}类原材料:")
            print(f"    供应商数量: {stats['supplier_count']} 家")
            print(f"    总订购量: {stats['total_orders']:,.0f} 立方米 ({ratio:.1f}%)")
            print(f"    平均周订购量: {stats['avg_weekly_order']:,.0f} 立方米")
            print(f"    最大周订购量: {stats['max_weekly_order']:,.0f} 立方米")
            print(f"    最小周订购量: {stats['min_weekly_order']:,.0f} 立方米")
            
            # 显示前5大供应商
            if len(stats['suppliers']) > 0:
                material_supplier_totals = analysis_df[
                    analysis_df['材料分类'] == material
                ].groupby('supplier_id')['order_amount'].sum().sort_values(ascending=False)
                
                print(f"    主要供应商 (前5名):")
                for i, (supplier_id, amount) in enumerate(material_supplier_totals.head(5).items()):
                    percentage = amount / stats['total_orders'] * 100
                    print(f"      {i+1}. {supplier_id}: {amount:,.0f} 立方米 ({percentage:.1f}%)")
        
        return material_stats
    
    def analyze_weekly_patterns(self, strategy_name, orders_df):
        """分析周度订购模式"""
        print(f"\n=== {strategy_name} 周度订购模式分析 ===")
        
        # 合并供应商信息
        analysis_df = orders_df.merge(
            self.supplier_data[['供应商ID', '材料分类']],
            left_on='supplier_id',
            right_on='供应商ID',
            how='left'
        )
        
        # 计算每周各材料的订购量
        weekly_materials = analysis_df.groupby(['week', '材料分类'])['order_amount'].sum().unstack(fill_value=0)
        
        print(f"\n前5周订购模式:")
        print(weekly_materials.head())
        
        print(f"\n各材料的周度统计:")
        for material in ['A', 'B', 'C']:
            if material in weekly_materials.columns:
                values = weekly_materials[material]
                print(f"  {material}类:")
                print(f"    平均: {values.mean():,.0f} 立方米")
                print(f"    标准差: {values.std():,.0f} 立方米")
                print(f"    变异系数: {values.std()/values.mean():.2f}")
        
        return weekly_materials
    
    def compare_strategies(self):
        """对比两种策略"""
        print(f"\n=== 策略对比分析 ===")
        
        # 分析两种策略
        stats1 = self.analyze_supplier_composition("策略1", self.strategy1_orders)
        stats2 = self.analyze_supplier_composition("策略2", self.strategy2_orders)
        
        # 对比表
        print(f"\n策略对比表:")
        print(f"{'指标':<20} {'策略1':<15} {'策略2':<15} {'策略2优势':<15}")
        print("-" * 70)
        
        for material in ['A', 'B', 'C']:
            total1 = stats1[material]['total_orders']
            total2 = stats2[material]['total_orders']
            
            ratio1 = total1 / sum([stats1[m]['total_orders'] for m in ['A', 'B', 'C']]) * 100
            ratio2 = total2 / sum([stats2[m]['total_orders'] for m in ['A', 'B', 'C']]) * 100
            
            improvement = ratio2 - ratio1
            improvement_text = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            
            print(f"{material}类比例{'':<14} {ratio1:.1f}%{'':<10} {ratio2:.1f}%{'':<10} {improvement_text}")
            print(f"{material}类供应商数{'':<10} {stats1[material]['supplier_count']}{'':<10} {stats2[material]['supplier_count']}{'':<10} {stats2[material]['supplier_count']-stats1[material]['supplier_count']:+d}")
    
    def create_visualization(self):
        """创建可视化图表"""
        print(f"\n=== 生成可视化图表 ===")
        
        # 分析两种策略
        stats1 = self.analyze_supplier_composition("策略1", self.strategy1_orders)
        stats2 = self.analyze_supplier_composition("策略2", self.strategy2_orders)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 材料比例对比
        materials = ['A', 'B', 'C']
        strategy1_ratios = []
        strategy2_ratios = []
        
        total1 = sum([stats1[m]['total_orders'] for m in materials])
        total2 = sum([stats2[m]['total_orders'] for m in materials])
        
        for material in materials:
            strategy1_ratios.append(stats1[material]['total_orders'] / total1 * 100)
            strategy2_ratios.append(stats2[material]['total_orders'] / total2 * 100)
        
        x = np.arange(len(materials))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, strategy1_ratios, width, label='策略1', alpha=0.8)
        axes[0, 0].bar(x + width/2, strategy2_ratios, width, label='策略2', alpha=0.8)
        axes[0, 0].set_xlabel('原材料类型')
        axes[0, 0].set_ylabel('比例 (%)')
        axes[0, 0].set_title('两种策略的材料比例对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(materials)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 供应商数量对比
        supplier_counts1 = [stats1[m]['supplier_count'] for m in materials]
        supplier_counts2 = [stats2[m]['supplier_count'] for m in materials]
        
        axes[0, 1].bar(x - width/2, supplier_counts1, width, label='策略1', alpha=0.8)
        axes[0, 1].bar(x + width/2, supplier_counts2, width, label='策略2', alpha=0.8)
        axes[0, 1].set_xlabel('原材料类型')
        axes[0, 1].set_ylabel('供应商数量')
        axes[0, 1].set_title('两种策略的供应商数量对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(materials)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 策略2的周度订购模式
        weekly_data2 = self.analyze_weekly_patterns("策略2", self.strategy2_orders)
        
        weeks = range(1, 25)
        for material in materials:
            if material in weekly_data2.columns:
                axes[1, 0].plot(weeks, weekly_data2[material], marker='o', label=f'{material}类', linewidth=2)
        
        axes[1, 0].set_xlabel('周数')
        axes[1, 0].set_ylabel('订购量 (立方米)')
        axes[1, 0].set_title('策略2: 24周订购模式')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 总订购量对比
        total_orders = [total1, total2]
        strategy_names = ['策略1', '策略2']
        
        bars = axes[1, 1].bar(strategy_names, total_orders, alpha=0.8, color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylabel('总订购量 (立方米)')
        axes[1, 1].set_title('两种策略的总订购量对比')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, total_orders):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/problem3_strategy_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 图表已保存到 results/problem3_strategy_comparison.png")
        
        plt.show()
    
    def generate_supplier_selection_report(self):
        """生成供应商选择报告"""
        print(f"\n=== 生成供应商选择报告 ===")
        
        # 分析策略2（推荐策略）
        analysis_df = self.strategy2_orders.merge(
            self.supplier_data[['供应商ID', '材料分类', '平均周制造能力', '最大周制造能力']],
            left_on='supplier_id',
            right_on='供应商ID',
            how='left'
        )
        
        # 按材料类型生成报告
        report_file = 'results/problem3_supplier_selection_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("问题三：供应商选择与供货量分配报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("一、总体概况\n")
            f.write("-"*30 + "\n")
            total_suppliers = analysis_df['supplier_id'].nunique()
            total_orders = analysis_df['order_amount'].sum()
            f.write(f"选定供应商总数: {total_suppliers} 家\n")
            f.write(f"24周总订购量: {total_orders:,.0f} 立方米\n")
            f.write(f"平均周订购量: {total_orders/24:,.0f} 立方米\n\n")
            
            # 按材料类型详细分析
            for material in ['A', 'B', 'C']:
                material_df = analysis_df[analysis_df['材料分类'] == material]
                
                if len(material_df) > 0:
                    f.write(f"二、{material}类原材料供应商配置\n")
                    f.write("-"*30 + "\n")
                    
                    material_suppliers = material_df['supplier_id'].nunique()
                    material_total = material_df['order_amount'].sum()
                    material_ratio = material_total / total_orders * 100
                    
                    f.write(f"供应商数量: {material_suppliers} 家\n")
                    f.write(f"总订购量: {material_total:,.0f} 立方米 ({material_ratio:.1f}%)\n")
                    f.write(f"平均周订购量: {material_total/24:,.0f} 立方米\n\n")
                    
                    # 供应商明细
                    supplier_details = material_df.groupby('supplier_id').agg({
                        'order_amount': ['sum', 'mean', 'count']
                    }).round(0)
                    supplier_details.columns = ['总订购量', '平均周订购量', '订购周数']
                    supplier_details = supplier_details.sort_values('总订购量', ascending=False)
                    
                    f.write(f"{material}类供应商明细表:\n")
                    f.write(f"{'供应商ID':<10} {'总订购量':<12} {'平均周订购量':<15} {'订购周数':<10} {'占比':<10}\n")
                    f.write("-"*70 + "\n")
                    
                    for supplier_id, row in supplier_details.head(20).iterrows():  # 显示前20名
                        percentage = row['总订购量'] / material_total * 100
                        f.write(f"{supplier_id:<10} {row['总订购量']:>10,.0f} {row['平均周订购量']:>13,.0f} {row['订购周数']:>8,.0f} {percentage:>8.1f}%\n")
                    
                    f.write("\n")
        
        print(f"✓ 供应商选择报告已保存到 {report_file}")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("="*60)
        print("问题三结果分析报告")
        print("="*60)
        
        # 加载数据
        self.load_data()
        
        # 对比策略
        self.compare_strategies()
        
        # 生成可视化
        self.create_visualization()
        
        # 生成供应商选择报告
        self.generate_supplier_selection_report()
        
        print("\n" + "="*60)
        print("分析完成！所有结果已保存到 results/ 目录")
        print("="*60)


def main():
    """主函数"""
    analyzer = Problem3ResultAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
