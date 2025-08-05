"""
第二问测试脚本
快速测试供应商和转运商分配方案
"""

from problem2_supplier_transporter_allocation import SupplierTransporterAllocator

def test_allocation():
    """测试分配功能"""
    print("第二问：供应商和转运商分配方案测试")
    print("="*60)
    
    # 创建分配器
    allocator = SupplierTransporterAllocator()
    
    # 设置较小的供应商数量进行快速测试
    allocator.set_supplier_count(150)
    
    try:
        # 加载数据
        allocator.load_data()
        
        # 生成供货计划
        allocator.generate_optimal_supply_plan()
        
        # 分配转运商
        allocator.allocate_transporters()
        
        # 导出结果
        allocator.export_results("test_allocation")
        
        print("\n测试完成！结果已导出到 results/ 目录")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_allocation()
