#!/usr/bin/env python3
"""
产能计算方法对比分析
对比80%最大产能方法与90%分位数方法的差异
"""

import pandas as pd

def compare_capacity_methods():
    """对比两种产能计算方法"""
    
    print("="*60)
    print("产能计算方法对比分析")
    print("="*60)
    
    # 原材料转换比例
    material_conversion = {'A': 0.6, 'B': 0.66, 'C': 0.78}
    
    # 读取数据
    capacity_df = pd.read_excel('DataFrames/供应商产品制造能力汇总.xlsx')
    percentile_df = pd.read_excel('DataFrames/供应商统计数据离散系数_重处理.xlsx')
    
    print("\n1. 原材料供应能力对比:")
    print("-" * 40)
    
    results = {}
    
    for material in ['A', 'B', 'C']:
        # 方法1：80%最大产能
        material_data = capacity_df[capacity_df['材料分类'] == material]
        max_capacity_80 = material_data['最大周制造能力'].sum() * 0.8
        
        # 方法2：90%分位数
        percentile_90 = percentile_df[percentile_df['材料分类'] == material]['90%分位数'].sum()
        
        # 计算差异
        reduction = (max_capacity_80 - percentile_90) / max_capacity_80 * 100
        
        print(f"{material}类材料:")
        print(f"  80%最大产能方法: {max_capacity_80:>10,.0f} 立方米/周")
        print(f"  90%分位数方法:   {percentile_90:>10,.0f} 立方米/周")
        print(f"  保守程度提升:    {reduction:>10.1f}%")
        print()
        
        results[material] = {
            'max_80': max_capacity_80,
            'percentile_90': percentile_90,
            'reduction': reduction
        }
    
    print("2. 产品制造能力对比:")
    print("-" * 40)
    
    for material in ['A', 'B', 'C']:
        # 转换为产品制造能力
        product_capacity_80 = results[material]['max_80'] / material_conversion[material]
        product_capacity_90 = results[material]['percentile_90'] / material_conversion[material]
        
        print(f"{material}类材料支持的产品制造能力:")
        print(f"  80%最大产能方法: {product_capacity_80:>10,.0f} 立方米/周")
        print(f"  90%分位数方法:   {product_capacity_90:>10,.0f} 立方米/周")
        print(f"  转换比例:        {material_conversion[material]:>10.2f}")
        print()
    
    print("3. 整体产能限制对比:")
    print("-" * 40)
    
    # 计算最终产能限制（受最小材料类型限制）
    product_capacities_80 = []
    product_capacities_90 = []
    
    for material in ['A', 'B', 'C']:
        product_cap_80 = results[material]['max_80'] / material_conversion[material]
        product_cap_90 = results[material]['percentile_90'] / material_conversion[material]
        product_capacities_80.append(product_cap_80)
        product_capacities_90.append(product_cap_90)
    
    final_capacity_80 = min(product_capacities_80)
    final_capacity_90 = min(product_capacities_90)
    
    print(f"最终产能限制（受最小材料约束）:")
    print(f"  80%最大产能方法: {final_capacity_80:>10,.0f} 立方米/周")
    print(f"  90%分位数方法:   {final_capacity_90:>10,.0f} 立方米/周")
    print(f"  保守程度提升:    {(final_capacity_80-final_capacity_90)/final_capacity_80*100:>10.1f}%")
    
    print("\n4. 方法论分析:")
    print("-" * 40)
    print("80%最大产能方法:")
    print("  - 基于供应商理论最大制造能力")
    print("  - 应用80%安全系数")
    print("  - 偏向乐观估计")
    print()
    print("90%分位数方法:")
    print("  - 基于供应商历史实际表现统计")
    print("  - 反映90%时间内的稳定供应能力")
    print("  - 更加现实和保守")
    print("  - 考虑了供应商产能的自然波动性")
    
    print("\n5. 建议:")
    print("-" * 40)
    print("推荐使用90%分位数方法，因为:")
    print("1. 基于历史数据，更贴近实际情况")
    print("2. 考虑了供应商产能的波动性和不确定性")
    print("3. 提供更稳健的供应链规划基础")
    print("4. 降低了因产能高估导致的供应风险")
    
    print("="*60)

if __name__ == "__main__":
    compare_capacity_methods()
