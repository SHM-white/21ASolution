"""
演示增强版ML模型的新功能
包括：模型保存/加载、多线程预测、进度条显示、GPU检测
"""

import numpy as np
import time
from supplier_prediction_model_v2 import get_trained_model, predict_multiple_suppliers

def demo_enhanced_features():
    """演示所有新功能"""
    print("🚀 增强版ML预测模型功能演示")
    print("=" * 60)
    
    # 1. 演示模型自动加载
    print("\n1️⃣ 模型自动加载功能")
    print("-" * 30)
    model = get_trained_model()
    print(f"✓ 模型训练状态: {model.is_trained}")
    print(f"✓ 模型保存路径: {model.model_file}")
    print(f"✓ GPU支持检测: {'已启用' if model.use_gpu else '未检测到GPU，使用CPU'}")
    
    # 2. 演示tqdm进度条和多线程预测
    print("\n2️⃣ 多线程批量预测 + 进度条")
    print("-" * 30)
    test_suppliers = ['S001', 'S002', 'S003', 'S004', 'S005', 'S010', 'S015', 'S020']
    
    # 单线程 vs 多线程性能对比
    print("🔄 单线程预测测试...")
    start_time = time.time()
    results_single = predict_multiple_suppliers(test_suppliers, 12, use_multithread=False)
    single_time = time.time() - start_time
    
    print(f"📊 多线程预测测试...")
    start_time = time.time()
    results_multi = predict_multiple_suppliers(test_suppliers, 12, use_multithread=True)
    multi_time = time.time() - start_time
    
    print(f"\n⚡ 性能对比:")
    print(f"   单线程时间: {single_time:.2f}秒")
    print(f"   多线程时间: {multi_time:.2f}秒")
    print(f"   性能提升: {single_time/multi_time:.2f}x")
    
    # 3. 演示预测结果质量
    print("\n3️⃣ 预测结果质量分析")
    print("-" * 30)
    for supplier_id in test_suppliers[:3]:
        predictions = results_multi[supplier_id]
        print(f"供应商 {supplier_id}:")
        print(f"  预测均值: {np.mean(predictions):.2f}")
        print(f"  预测变异系数: {np.std(predictions)/np.mean(predictions)*100:.1f}%")
        print(f"  预测范围: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
    
    # 4. 演示模型保存功能
    print("\n4️⃣ 模型持久化功能")
    print("-" * 30)
    save_success = model.save_model()
    print(f"✓ 模型保存结果: {'成功' if save_success else '失败'}")
    
    # 验证模型可以重新加载
    from supplier_prediction_model_v2 import SupplierPredictionModel
    new_model = SupplierPredictionModel()
    load_success = new_model.load_model()
    print(f"✓ 模型重新加载: {'成功' if load_success else '失败'}")
    print(f"✓ 模型一致性检查: {'通过' if new_model.is_trained else '失败'}")
    
    # 5. 演示大规模预测能力
    print("\n5️⃣ 大规模预测能力测试")
    print("-" * 30)
    large_supplier_list = [f'S{i:03d}' for i in range(1, 51)]  # 50个供应商
    print(f"📈 测试50个供应商的24周预测...")
    
    start_time = time.time()
    large_results = predict_multiple_suppliers(large_supplier_list, 24, use_multithread=True)
    large_time = time.time() - start_time
    
    total_predictions = sum(len(pred) for pred in large_results.values())
    print(f"✓ 预测完成时间: {large_time:.2f}秒")
    print(f"✓ 总预测次数: {total_predictions:,}")
    print(f"✓ 预测速度: {total_predictions/large_time:.0f} 预测/秒")
    
    print("\n🎉 所有功能演示完成!")
    print("=" * 60)
    
    return {
        'single_thread_time': single_time,
        'multi_thread_time': multi_time,
        'speedup': single_time/multi_time,
        'large_scale_time': large_time,
        'predictions_per_second': total_predictions/large_time
    }

if __name__ == "__main__":
    # 运行演示
    results = demo_enhanced_features()
    
    print(f"\n📋 性能总结:")
    print(f"多线程加速比: {results['speedup']:.2f}x")
    print(f"大规模预测速度: {results['predictions_per_second']:.0f} 预测/秒")
