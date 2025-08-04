"""
转运商损耗率分析与ARIMA预测模型
基于历史损耗率数据，分析转运商表现并预测未来损耗率
支持多种评估算法和优化策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import combinations
import warnings
import os
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['黑体', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TransporterLossAnalyzer:
    """转运商损耗率分析器"""
    
    def __init__(self, data_file_path=None):
        """
        初始化转运商损耗率分析器
        
        Args:
            data_file_path: 转运商数据文件路径
        """
        self.data_file_path = data_file_path or 'C/附件2 近5年8家转运商的相关数据.xlsx'
        self.transporter_data = None
        self.transporter_names = None
        self.loss_data = None
        self.transporter_analysis = {}
        self.arima_models = {}
        self.predictions = {}
        
        # 创建必要的文件夹
        for folder in ['log', 'Pictures', 'models']:
            os.makedirs(folder, exist_ok=True)
            
        # 设置日志
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'log/transporter_analysis_{timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """加载转运商损耗率数据"""
        try:
            self.logger.info("开始加载转运商损耗率数据...")
            
            # 读取转运商损耗率数据
            self.transporter_data = pd.read_excel(
                self.data_file_path, 
                sheet_name='运输损耗率（%）'
            )
            
            # 提取转运商名称和损耗率数据
            self.transporter_names = self.transporter_data.iloc[:, 0].values
            self.loss_data = self.transporter_data.iloc[:, 1:].values
            
            self.logger.info(f"成功加载数据: {len(self.transporter_names)}家转运商, {self.loss_data.shape[1]}周数据")
            
            # 数据基本信息
            print(f"转运商数量: {len(self.transporter_names)}")
            print(f"数据周数: {self.loss_data.shape[1]}")
            print(f"转运商列表: {list(self.transporter_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            return False
    
    def analyze_transporter_characteristics(self):
        """分析转运商损耗率特征"""
        if self.loss_data is None:
            self.logger.error("请先加载数据")
            return None
            
        self.logger.info("开始分析转运商损耗率特征...")
        
        transporter_metrics = []
        
        for i, transporter_name in enumerate(self.transporter_names):
            # 获取该转运商的损耗率数据
            loss_rates = self.loss_data[i]
            
            # 过滤有效数据（去除0值和异常值）
            valid_loss_rates = loss_rates[
                (loss_rates > 0) & 
                (loss_rates < 100) & 
                (~np.isnan(loss_rates))
            ]
            
            if len(valid_loss_rates) == 0:
                self.logger.warning(f"转运商 {transporter_name} 无有效数据")
                continue
                
            # 计算基础统计指标
            metrics = {
                'transporter_name': transporter_name,
                'total_weeks': len(loss_rates),
                'valid_weeks': len(valid_loss_rates),
                'avg_loss_rate': np.mean(valid_loss_rates),
                'median_loss_rate': np.median(valid_loss_rates),
                'std_loss_rate': np.std(valid_loss_rates),
                'min_loss_rate': np.min(valid_loss_rates),
                'max_loss_rate': np.max(valid_loss_rates),
                'cv_loss_rate': np.std(valid_loss_rates) / np.mean(valid_loss_rates) if np.mean(valid_loss_rates) > 0 else 0,
                'q25_loss_rate': np.percentile(valid_loss_rates, 25),
                'q75_loss_rate': np.percentile(valid_loss_rates, 75),
                'data_completeness': len(valid_loss_rates) / len(loss_rates)
            }
            
            # 计算趋势指标
            if len(valid_loss_rates) > 1:
                # 线性趋势
                x = np.arange(len(valid_loss_rates))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, valid_loss_rates)
                metrics['trend_slope'] = slope
                metrics['trend_r_squared'] = r_value ** 2
                metrics['trend_p_value'] = p_value
                
                # 近期表现（最后1/4数据）
                recent_data_size = max(1, len(valid_loss_rates) // 4)
                recent_loss_rates = valid_loss_rates[-recent_data_size:]
                metrics['recent_avg_loss_rate'] = np.mean(recent_loss_rates)
                metrics['recent_vs_overall'] = metrics['recent_avg_loss_rate'] - metrics['avg_loss_rate']
            else:
                metrics.update({
                    'trend_slope': 0,
                    'trend_r_squared': 0,
                    'trend_p_value': 1,
                    'recent_avg_loss_rate': metrics['avg_loss_rate'],
                    'recent_vs_overall': 0
                })
            
            # 计算稳定性得分（损耗率越低越好，稳定性越高越好）
            stability_score = 100 / (1 + metrics['cv_loss_rate'])
            efficiency_score = 100 - metrics['avg_loss_rate']
            comprehensive_score = 0.6 * efficiency_score + 0.4 * stability_score
            
            metrics.update({
                'stability_score': stability_score,
                'efficiency_score': efficiency_score,
                'comprehensive_score': comprehensive_score
            })
            
            transporter_metrics.append(metrics)
            self.transporter_analysis[transporter_name] = metrics
        
        # 转换为DataFrame
        self.analysis_df = pd.DataFrame(transporter_metrics)
        
        # 排名分析
        self.analysis_df['avg_loss_rank'] = self.analysis_df['avg_loss_rate'].rank(ascending=True)
        self.analysis_df['stability_rank'] = self.analysis_df['stability_score'].rank(ascending=False)
        self.analysis_df['comprehensive_rank'] = self.analysis_df['comprehensive_score'].rank(ascending=False)
        
        self.logger.info(f"完成 {len(transporter_metrics)} 家转运商的特征分析")
        
        # 输出分析结果
        print("\\n转运商损耗率分析结果:")
        print("=" * 60)
        print(f"分析转运商数量: {len(transporter_metrics)}")
        print(f"平均损耗率: {self.analysis_df['avg_loss_rate'].mean():.3f}%")
        print(f"损耗率范围: {self.analysis_df['min_loss_rate'].min():.3f}% - {self.analysis_df['max_loss_rate'].max():.3f}%")
        print(f"损耗率标准差: {self.analysis_df['avg_loss_rate'].std():.3f}%")
        
        return self.analysis_df
    
    def get_transporter_ranking(self, sort_by='comprehensive_score'):
        """获取转运商排名"""
        if self.analysis_df is None:
            self.logger.error("请先进行转运商特征分析")
            return None
            
        # 按指定指标排序
        if sort_by == 'avg_loss_rate':
            ranking = self.analysis_df.sort_values('avg_loss_rate', ascending=True)
        else:
            ranking = self.analysis_df.sort_values(sort_by, ascending=False)
        
        print(f"\\n转运商排名 (按{sort_by}排序):")
        print("=" * 80)
        
        for idx, (_, transporter) in enumerate(ranking.iterrows(), 1):
            print(f"{idx:2d}. {transporter['transporter_name']}")
            print(f"     平均损耗率: {transporter['avg_loss_rate']:.3f}%")
            print(f"     损耗率范围: {transporter['min_loss_rate']:.3f}% - {transporter['max_loss_rate']:.3f}%")
            print(f"     稳定性得分: {transporter['stability_score']:.2f}")
            print(f"     综合得分: {transporter['comprehensive_score']:.2f}")
            print(f"     数据完整性: {transporter['data_completeness']:.2f}")
            
        return ranking
    
    def visualize_transporter_analysis(self):
        """可视化转运商分析结果"""
        if self.analysis_df is None:
            self.logger.error("请先进行转运商特征分析")
            return
            
        self.logger.info("开始生成转运商分析可视化图表...")
        
        # 1. 转运商平均损耗率对比
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(self.analysis_df)))
        bars = plt.bar(self.analysis_df['transporter_name'], 
                      self.analysis_df['avg_loss_rate'], 
                      color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, self.analysis_df['avg_loss_rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('转运商平均损耗率对比', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('转运商', fontsize=14)
        plt.ylabel('平均损耗率 (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('Pictures/transporter_avg_loss_comparison.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 转运商综合评分散点图
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(self.analysis_df['avg_loss_rate'], 
                            self.analysis_df['stability_score'],
                            s=self.analysis_df['comprehensive_score']*3,
                            c=self.analysis_df['comprehensive_score'], 
                            cmap='RdYlGn', alpha=0.7)
        
        # 添加转运商名称标签
        for _, row in self.analysis_df.iterrows():
            plt.annotate(row['transporter_name'], 
                        (row['avg_loss_rate'], row['stability_score']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.colorbar(scatter, label='综合得分')
        plt.title('转运商损耗率vs稳定性分析\\n(气泡大小表示综合得分)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('平均损耗率 (%)', fontsize=14)
        plt.ylabel('稳定性得分', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Pictures/transporter_loss_vs_stability.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 转运商损耗率分布热力图
        plt.figure(figsize=(16, 10))
        
        # 创建热力图数据
        heatmap_data = []
        weeks = list(range(1, self.loss_data.shape[1] + 1))
        
        for i, name in enumerate(self.transporter_names):
            # 填充损耗率数据，0值用NaN表示
            loss_rates = self.loss_data[i].copy()
            loss_rates[loss_rates == 0] = np.nan
            heatmap_data.append(loss_rates)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=self.transporter_names,
                                 columns=[f'第{w}周' for w in weeks])
        
        # 绘制热力图
        sns.heatmap(heatmap_df, 
                   cmap='RdYlGn_r', 
                   annot=False,
                   fmt='.1f',
                   cbar_kws={'label': '损耗率 (%)'},
                   xticklabels=10,  # 每10周显示一个标签
                   yticklabels=True)
        
        plt.title('转运商周损耗率热力图', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('周数', fontsize=14)
        plt.ylabel('转运商', fontsize=14)
        plt.tight_layout()
        plt.savefig('Pictures/transporter_weekly_loss_heatmap.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("转运商分析可视化图表生成完成")
    
    def prepare_time_series_data(self, transporter_name):
        """为指定转运商准备时间序列数据"""
        if transporter_name not in self.transporter_analysis:
            self.logger.error(f"转运商 {transporter_name} 不存在")
            return None
            
        # 获取转运商索引
        transporter_idx = list(self.transporter_names).index(transporter_name)
        
        # 获取损耗率数据
        loss_rates = self.loss_data[transporter_idx]
        
        # 数据预处理
        # 1. 处理缺失值和异常值
        processed_data = []
        for rate in loss_rates:
            if rate <= 0 or rate >= 100 or np.isnan(rate):
                # 用前一个有效值填充，如果没有则用后续值
                if len(processed_data) > 0:
                    processed_data.append(processed_data[-1])
                else:
                    # 寻找下一个有效值
                    next_valid = None
                    for future_rate in loss_rates[len(processed_data):]:
                        if 0 < future_rate < 100 and not np.isnan(future_rate):
                            next_valid = future_rate
                            break
                    processed_data.append(next_valid if next_valid is not None else 2.0)
            else:
                processed_data.append(rate)
        
        # 2. 平滑处理（移动平均）
        window_size = min(3, len(processed_data))
        if window_size > 1:
            smoothed_data = np.convolve(processed_data, 
                                       np.ones(window_size)/window_size, 
                                       mode='same')
        else:
            smoothed_data = processed_data
        
        return np.array(smoothed_data)
    
    def check_stationarity(self, data, transporter_name):
        """检查时间序列的平稳性"""
        self.logger.info(f"检查 {transporter_name} 的时间序列平稳性...")
        
        # ADF检验
        adf_result = adfuller(data)
        
        print(f"\\n{transporter_name} 平稳性检验结果:")
        print(f"ADF统计量: {adf_result[0]:.6f}")
        print(f"p值: {adf_result[1]:.6f}")
        print(f"临界值:")
        for key, value in adf_result[4].items():
            print(f"  {key}: {value:.6f}")
        
        is_stationary = adf_result[1] < 0.05
        print(f"是否平稳: {'是' if is_stationary else '否'}")
        
        return is_stationary, adf_result
    
    def difference_series(self, data, order=1):
        """对时间序列进行差分"""
        for i in range(order):
            data = np.diff(data)
        return data
    
    def find_optimal_arima_params(self, data, transporter_name, max_p=3, max_d=2, max_q=3):
        """寻找最优ARIMA参数"""
        self.logger.info(f"为 {transporter_name} 寻找最优ARIMA参数...")
        
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        results = []
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        aic = fitted_model.aic
                        bic = fitted_model.bic
                        
                        results.append({
                            'p': p, 'd': d, 'q': q,
                            'aic': aic, 'bic': bic,
                            'model': fitted_model
                        })
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                            best_model = fitted_model
                            
                    except Exception as e:
                        continue
        
        if best_params:
            self.logger.info(f"{transporter_name} 最优ARIMA参数: {best_params}, AIC: {best_aic:.4f}")
        else:
            self.logger.warning(f"{transporter_name} 未找到合适的ARIMA参数")
            
        return best_params, best_model, results
    
    def build_arima_model(self, transporter_name, forecast_weeks=24):
        """为指定转运商构建ARIMA预测模型"""
        self.logger.info(f"为 {transporter_name} 构建ARIMA预测模型...")
        
        # 准备时间序列数据
        data = self.prepare_time_series_data(transporter_name)
        if data is None:
            return None
        
        # 检查平稳性
        is_stationary, adf_result = self.check_stationarity(data, transporter_name)
        
        # 寻找最优参数
        best_params, best_model, all_results = self.find_optimal_arima_params(data, transporter_name)
        
        if best_model is None:
            self.logger.error(f"{transporter_name} ARIMA模型构建失败")
            return None
        
        # 进行预测
        forecast_result = best_model.forecast(steps=forecast_weeks)
        forecast_conf_int = best_model.get_forecast(steps=forecast_weeks).conf_int()
        
        # 计算预测精度指标
        fitted_values = best_model.fittedvalues
        residuals = best_model.resid
        
        # 样本内评估 - 确保数据长度一致
        # 对于差分模型，fitted_values长度会减少
        if len(fitted_values) == len(data) - 1:
            # 差分模型，跳过第一个观测值
            actual_data = data[1:]
        else:
            # 非差分模型，使用全部数据
            actual_data = data[:len(fitted_values)]
        
        mae = mean_absolute_error(actual_data, fitted_values)
        mse = mean_squared_error(actual_data, fitted_values)
        rmse = np.sqrt(mse)
        
        model_info = {
            'transporter_name': transporter_name,
            'best_params': best_params,
            'model': best_model,
            'forecast': forecast_result,
            'forecast_conf_int': forecast_conf_int,
            'original_data': data,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'all_results': all_results
        }
        
        self.arima_models[transporter_name] = model_info
        
        print(f"\\n{transporter_name} ARIMA模型结果:")
        print(f"最优参数: ARIMA{best_params}")
        print(f"AIC: {best_model.aic:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"预测未来{forecast_weeks}周的平均损耗率: {np.mean(forecast_result):.3f}%")
        
        return model_info
    
    def build_all_arima_models(self, forecast_weeks=24):
        """为所有转运商构建ARIMA预测模型"""
        self.logger.info("开始为所有转运商构建ARIMA预测模型...")
        
        successful_models = 0
        
        for transporter_name in self.transporter_names:
            try:
                model_info = self.build_arima_model(transporter_name, forecast_weeks)
                if model_info is not None:
                    successful_models += 1
            except Exception as e:
                self.logger.error(f"{transporter_name} 模型构建失败: {e}")
                continue
        
        self.logger.info(f"成功构建 {successful_models}/{len(self.transporter_names)} 个ARIMA模型")
        
        return successful_models
    
    def visualize_arima_results(self, transporter_name):
        """可视化ARIMA模型结果"""
        if transporter_name not in self.arima_models:
            self.logger.error(f"{transporter_name} 的ARIMA模型不存在")
            return
        
        model_info = self.arima_models[transporter_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{transporter_name} ARIMA模型分析结果', fontsize=16, fontweight='bold')
        
        # 1. 原始数据和拟合值
        ax1 = axes[0, 0]
        original_data = model_info['original_data']
        fitted_values = model_info['fitted_values']
        
        ax1.plot(original_data, label='原始数据', color='blue', linewidth=2)
        ax1.plot(range(1, len(fitted_values)+1), fitted_values, 
                label='拟合值', color='red', linestyle='--', linewidth=2)
        ax1.set_title('原始数据 vs 拟合值')
        ax1.set_xlabel('周数')
        ax1.set_ylabel('损耗率 (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测结果
        ax2 = axes[0, 1]
        forecast = model_info['forecast']
        conf_int = model_info['forecast_conf_int']
        
        # 绘制历史数据
        historical_weeks = range(len(original_data))
        ax2.plot(historical_weeks, original_data, label='历史数据', color='blue', linewidth=2)
        
        # 绘制预测数据
        forecast_weeks = range(len(original_data), len(original_data) + len(forecast))
        ax2.plot(forecast_weeks, forecast, label='预测值', color='red', linewidth=2)
        
        # 处理置信区间数据
        if hasattr(conf_int, 'iloc'):
            # 如果是DataFrame
            lower_bound = conf_int.iloc[:, 0]
            upper_bound = conf_int.iloc[:, 1]
        else:
            # 如果是numpy数组
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
        
        ax2.fill_between(forecast_weeks, 
                        lower_bound, upper_bound,
                        alpha=0.3, color='red', label='95%置信区间')
        
        ax2.set_title('损耗率预测')
        ax2.set_xlabel('周数')
        ax2.set_ylabel('损耗率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差分析
        ax3 = axes[1, 0]
        residuals = model_info['residuals']
        ax3.plot(residuals, color='green', linewidth=1)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_title('残差序列')
        ax3.set_xlabel('周数')
        ax3.set_ylabel('残差')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差分布
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=15, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        
        # 添加正态分布曲线
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='正态分布拟合')
        
        ax4.set_title('残差分布')
        ax4.set_xlabel('残差值')
        ax4.set_ylabel('密度')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'Pictures/arima_analysis_{transporter_name}.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出模型评估信息
        print(f"\\n{transporter_name} ARIMA模型评估:")
        print(f"模型参数: ARIMA{model_info['best_params']}")
        print(f"AIC: {model_info['aic']:.4f}")
        print(f"BIC: {model_info['bic']:.4f}")
        print(f"MAE: {model_info['mae']:.4f}")
        print(f"RMSE: {model_info['rmse']:.4f}")
        print(f"残差均值: {np.mean(residuals):.6f}")
        print(f"残差标准差: {np.std(residuals):.4f}")
    
    def calculate_transport_capacity_requirements(self, material_type='C', planning_weeks=24):
        """
        计算全部订购C类材料所需的转运能力
        
        Args:
            material_type: 材料类型，默认'C'
            planning_weeks: 规划周数，默认24周
        """
        self.logger.info(f"计算全部订购{material_type}类材料的转运能力需求...")
        
        # 生产参数
        weekly_production_capacity = 28200  # 每周产能 (m³)
        material_conversion_rates = {
            'A': 0.6,   # A类材料消耗量系数
            'B': 0.66,  # B类材料消耗量系数  
            'C': 0.72   # C类材料消耗量系数
        }
        
        # 计算每周原材料需求量
        conversion_rate = material_conversion_rates[material_type]
        weekly_material_demand = weekly_production_capacity * conversion_rate
        
        # 计算总需求量（考虑库存积累）
        total_demand = weekly_material_demand * planning_weeks
        
        # 考虑安全库存（10%）
        safety_stock_factor = 1.1
        total_demand_with_safety = total_demand * safety_stock_factor
        
        print(f"\\n{material_type}类材料转运需求分析:")
        print(f"每周产能需求: {weekly_production_capacity:,.0f} m³")
        print(f"{material_type}类材料转换系数: {conversion_rate}")
        print(f"每周{material_type}类材料需求: {weekly_material_demand:,.0f} m³")
        print(f"{planning_weeks}周总需求: {total_demand:,.0f} m³")
        print(f"含安全库存总需求: {total_demand_with_safety:,.0f} m³")
        
        return {
            'material_type': material_type,
            'weekly_production_capacity': weekly_production_capacity,
            'conversion_rate': conversion_rate,
            'weekly_material_demand': weekly_material_demand,
            'planning_weeks': planning_weeks,
            'total_demand': total_demand,
            'total_demand_with_safety': total_demand_with_safety
        }
    
    def optimize_transporter_combination(self, material_type='C', algorithm='min_loss'):
        """
        优化转运商组合选择（考虑单周运力限制）
        
        Args:
            material_type: 材料类型，默认'C'
            algorithm: 优化算法 ('min_loss', 'weighted_score', 'capacity_efficiency')
        """
        if not self.arima_models:
            self.logger.error("请先构建ARIMA预测模型")
            return None
            
        self.logger.info(f"使用{algorithm}算法优化转运商组合选择...")
        
        # 计算单周转运需求
        weekly_production_capacity = 28200  # 每周产能 (m³)
        material_conversion_rates = {
            'A': 0.6, 'B': 0.66, 'C': 0.72
        }
        
        conversion_rate = material_conversion_rates[material_type]
        weekly_material_demand = weekly_production_capacity * conversion_rate
        
        # 转运商单周运力限制
        transporter_capacity_limit = 6000  # 每家转运商单周运力限制 (m³)
        
        print(f"\\n{material_type}类材料单周转运需求分析:")
        print(f"每周产能需求: {weekly_production_capacity:,.0f} m³")
        print(f"{material_type}类材料转换系数: {conversion_rate}")
        print(f"每周{material_type}类材料需求: {weekly_material_demand:,.0f} m³")
        print(f"转运商单周运力限制: {transporter_capacity_limit:,.0f} m³/周")
        
        # 计算需要的最少转运商数量
        min_transporters_needed = int(np.ceil(weekly_material_demand / transporter_capacity_limit))
        print(f"理论最少需要转运商数量: {min_transporters_needed} 家")
        
        # 收集所有转运商的预测数据
        transporter_data = {}
        
        for transporter_name, model_info in self.arima_models.items():
            forecast = model_info['forecast']
            avg_predicted_loss = np.mean(forecast)
            effective_capacity_rate = (100 - avg_predicted_loss) / 100
            
            # 考虑损耗后的有效运力
            effective_capacity = transporter_capacity_limit * effective_capacity_rate
            
            analysis_data = self.transporter_analysis[transporter_name]
            
            transporter_data[transporter_name] = {
                'predicted_loss_rate': avg_predicted_loss,
                'effective_capacity_rate': effective_capacity_rate,
                'effective_capacity': effective_capacity,
                'max_capacity': transporter_capacity_limit,
                'stability_score': analysis_data['stability_score'],
                'comprehensive_score': analysis_data['comprehensive_score'],
                'data_completeness': analysis_data['data_completeness']
            }
        
        # 生成所有可能的转运商组合
        
        best_combination = None
        best_metrics = None
        min_loss_rate = float('inf')
        
        # 尝试不同数量的转运商组合
        for num_transporters in range(min_transporters_needed, len(self.transporter_names) + 1):
            for combination in combinations(self.transporter_names, num_transporters):
                # 检查组合是否能满足运力需求
                total_effective_capacity = sum(transporter_data[t]['effective_capacity'] for t in combination)
                
                if total_effective_capacity >= weekly_material_demand:
                    # 计算组合的综合指标
                    combination_metrics = self._evaluate_transporter_combination(
                        combination, transporter_data, weekly_material_demand, algorithm
                    )
                    
                    # 根据算法选择最优组合
                    if algorithm == 'min_loss':
                        if combination_metrics['weighted_avg_loss_rate'] < min_loss_rate:
                            min_loss_rate = combination_metrics['weighted_avg_loss_rate']
                            best_combination = combination
                            best_metrics = combination_metrics
                    
                    elif algorithm == 'weighted_score':
                        if best_metrics is None or combination_metrics['weighted_score'] > best_metrics['weighted_score']:
                            best_combination = combination
                            best_metrics = combination_metrics
                    
                    elif algorithm == 'capacity_efficiency':
                        if best_metrics is None or combination_metrics['capacity_efficiency'] > best_metrics['capacity_efficiency']:
                            best_combination = combination
                            best_metrics = combination_metrics
        
        if best_combination is None:
            self.logger.error("未找到满足运力需求的转运商组合")
            return None
        
        # 计算最优组合的运输分配方案
        allocation_plan = self._calculate_optimal_allocation(
            best_combination, transporter_data, weekly_material_demand
        )
        
        # 输出结果
        self._print_combination_results(
            best_combination, best_metrics, allocation_plan, 
            material_type, weekly_material_demand, algorithm
        )
        
        return {
            'algorithm': algorithm,
            'best_combination': best_combination,
            'combination_metrics': best_metrics,
            'allocation_plan': allocation_plan,
            'material_type': material_type,
            'weekly_demand': weekly_material_demand,
            'transporter_data': transporter_data
        }
    
    def _evaluate_transporter_combination(self, combination, transporter_data, weekly_demand, algorithm):
        """评估转运商组合的指标"""
        total_capacity = sum(transporter_data[t]['max_capacity'] for t in combination)
        total_effective_capacity = sum(transporter_data[t]['effective_capacity'] for t in combination)
        
        # 计算加权平均损耗率（按运力权重）
        weighted_loss_rate = 0
        weighted_stability = 0
        weighted_comprehensive = 0
        
        for transporter in combination:
            data = transporter_data[transporter]
            weight = data['max_capacity'] / total_capacity
            
            weighted_loss_rate += data['predicted_loss_rate'] * weight
            weighted_stability += data['stability_score'] * weight
            weighted_comprehensive += data['comprehensive_score'] * weight
        
        # 运力效率（有效运力/需求量）
        capacity_efficiency = total_effective_capacity / weekly_demand
        
        # 运力利用率
        capacity_utilization = weekly_demand / total_effective_capacity if total_effective_capacity > 0 else 0
        
        # 综合得分
        if algorithm == 'weighted_score':
            weighted_score = (
                0.4 * (100 - weighted_loss_rate) +  # 损耗率得分
                0.3 * weighted_stability +          # 稳定性得分
                0.2 * weighted_comprehensive +      # 综合表现得分
                0.1 * (capacity_efficiency * 20)    # 运力效率得分
            )
        else:
            weighted_score = 0
        
        return {
            'num_transporters': len(combination),
            'total_capacity': total_capacity,
            'total_effective_capacity': total_effective_capacity,
            'weighted_avg_loss_rate': weighted_loss_rate,
            'weighted_stability': weighted_stability,
            'weighted_comprehensive': weighted_comprehensive,
            'capacity_efficiency': capacity_efficiency,
            'capacity_utilization': capacity_utilization,
            'weighted_score': weighted_score
        }
    
    def _calculate_optimal_allocation(self, combination, transporter_data, weekly_demand):
        """计算最优运输分配方案"""
        allocation_plan = {}
        
        # 按有效运力比例分配
        total_effective_capacity = sum(transporter_data[t]['effective_capacity'] for t in combination)
        
        for transporter in combination:
            data = transporter_data[transporter]
            
            # 按比例分配，但不超过单个转运商的运力限制
            proportional_allocation = (data['effective_capacity'] / total_effective_capacity) * weekly_demand
            
            # 考虑损耗，计算需要运输的原始量
            required_transport = proportional_allocation / data['effective_capacity_rate']
            
            # 确保不超过运力限制
            actual_transport = min(required_transport, data['max_capacity'])
            
            # 计算实际接收量（扣除损耗）
            actual_received = actual_transport * data['effective_capacity_rate']
            
            allocation_plan[transporter] = {
                'transport_volume': actual_transport,
                'received_volume': actual_received,
                'loss_volume': actual_transport - actual_received,
                'loss_rate': data['predicted_loss_rate'],
                'capacity_utilization': actual_transport / data['max_capacity']
            }
        
        return allocation_plan
    
    def _print_combination_results(self, combination, metrics, allocation_plan, material_type, weekly_demand, algorithm):
        """输出转运商组合结果"""
        print(f"\\n转运商组合优化结果 ({algorithm}算法):")
        print("=" * 80)
        print(f"最优转运商组合: {', '.join(combination)}")
        print(f"组合转运商数量: {metrics['num_transporters']} 家")
        print(f"总运力: {metrics['total_capacity']:,.0f} m³/周")
        print(f"有效运力: {metrics['total_effective_capacity']:,.0f} m³/周")
        print(f"加权平均损耗率: {metrics['weighted_avg_loss_rate']:.3f}%")
        print(f"运力效率: {metrics['capacity_efficiency']:.2f}")
        print(f"运力利用率: {metrics['capacity_utilization']:.1%}")
        
        if algorithm == 'weighted_score':
            print(f"加权综合得分: {metrics['weighted_score']:.2f}")
        
        print(f"\\n详细分配方案:")
        print("-" * 80)
        print(f"{'转运商':<8} {'运输量(m³)':<12} {'接收量(m³)':<12} {'损耗量(m³)':<12} {'损耗率(%)':<10} {'运力利用率':<10}")
        print("-" * 80)
        
        total_transport = 0
        total_received = 0
        total_loss = 0
        
        for transporter, plan in allocation_plan.items():
            print(f"{transporter:<8} {plan['transport_volume']:<12.0f} {plan['received_volume']:<12.0f} "
                  f"{plan['loss_volume']:<12.0f} {plan['loss_rate']:<10.3f} {plan['capacity_utilization']:<10.1%}")
            
            total_transport += plan['transport_volume']
            total_received += plan['received_volume']
            total_loss += plan['loss_volume']
        
        print("-" * 80)
        print(f"{'合计':<8} {total_transport:<12.0f} {total_received:<12.0f} "
              f"{total_loss:<12.0f} {(total_loss/total_transport*100) if total_transport > 0 else 0:<10.3f} {'--':<10}")
        
        print(f"\\n方案评估:")
        print(f"{material_type}类材料周需求: {weekly_demand:,.0f} m³")
        print(f"实际可接收量: {total_received:,.0f} m³")
        print(f"需求满足率: {(total_received/weekly_demand):.1%}")
        print(f"总损耗量: {total_loss:,.0f} m³")
        print(f"总损耗率: {(total_loss/total_transport*100) if total_transport > 0 else 0:.3f}%")
    
    def optimize_transporter_selection(self, material_type='C', planning_weeks=24, algorithm='weighted_score'):
        """
        优化转运商选择方案（原有的单转运商方法，保持兼容性）
        """
        # 调用新的组合优化方法
        return self.optimize_transporter_combination(material_type, algorithm)
    
    def analyze_cumulative_transporter_capacity(self, sort_by='predicted_loss_rate'):
        """
        分析转运商累计运力和损耗率期望
        
        Args:
            sort_by: 排序依据 ('predicted_loss_rate', 'comprehensive_score', 'stability_score')
        """
        if not self.arima_models:
            self.logger.error("请先构建ARIMA预测模型")
            return None
            
        self.logger.info("分析转运商累计运力和损耗率期望...")
        
        # 收集转运商预测数据
        transporter_summary = []
        
        for transporter_name, model_info in self.arima_models.items():
            forecast = model_info['forecast']
            avg_predicted_loss = np.mean(forecast)
            
            analysis_data = self.transporter_analysis[transporter_name]
            
            transporter_summary.append({
                'transporter_name': transporter_name,
                'predicted_loss_rate': avg_predicted_loss,
                'comprehensive_score': analysis_data['comprehensive_score'],
                'stability_score': analysis_data['stability_score'],
                'data_completeness': analysis_data['data_completeness'],
                'max_capacity': 6000,  # 每家转运商运力限制
                'effective_capacity_rate': (100 - avg_predicted_loss) / 100,
                'effective_capacity': 6000 * (100 - avg_predicted_loss) / 100
            })
        
        # 转换为DataFrame并排序
        summary_df = pd.DataFrame(transporter_summary)
        
        # 根据排序依据进行排序
        if sort_by == 'predicted_loss_rate':
            summary_df = summary_df.sort_values('predicted_loss_rate', ascending=True)
            sort_label = "预测损耗率（升序）"
        elif sort_by == 'comprehensive_score':
            summary_df = summary_df.sort_values('comprehensive_score', ascending=False)
            sort_label = "综合得分（降序）"
        elif sort_by == 'stability_score':
            summary_df = summary_df.sort_values('stability_score', ascending=False)
            sort_label = "稳定性得分（降序）"
        else:
            summary_df = summary_df.sort_values('predicted_loss_rate', ascending=True)
            sort_label = "预测损耗率（升序）"
        
        # 计算累计指标
        cumulative_analysis = []
        
        for i in range(1, len(summary_df) + 1):
            top_n = summary_df.head(i)
            
            # 累计运力
            total_capacity = i * 6000
            total_effective_capacity = top_n['effective_capacity'].sum()
            
            # 加权平均损耗率（按运力权重）
            weighted_loss_rate = (top_n['predicted_loss_rate'] * top_n['max_capacity']).sum() / total_capacity
            
            # 加权平均综合得分
            weighted_comprehensive = (top_n['comprehensive_score'] * top_n['max_capacity']).sum() / total_capacity
            
            # 加权平均稳定性得分
            weighted_stability = (top_n['stability_score'] * top_n['max_capacity']).sum() / total_capacity
            
            # 数据完整性平均值
            avg_completeness = top_n['data_completeness'].mean()
            
            cumulative_analysis.append({
                'num_transporters': i,
                'transporter_list': ', '.join(top_n['transporter_name'].tolist()),
                'total_capacity': total_capacity,
                'total_effective_capacity': total_effective_capacity,
                'weighted_loss_rate': weighted_loss_rate,
                'effective_capacity_rate': total_effective_capacity / total_capacity,
                'weighted_comprehensive_score': weighted_comprehensive,
                'weighted_stability_score': weighted_stability,
                'avg_data_completeness': avg_completeness
            })
        
        cumulative_df = pd.DataFrame(cumulative_analysis)
        
        # 输出结果表格
        print(f"\\n转运商累计运力分析表格 (排序依据: {sort_label})")
        print("=" * 150)
        print(f"{'前N家':<6} {'转运商组合':<35} {'总运力':<10} {'有效运力':<10} {'损耗率期望':<10} {'有效率':<8} {'综合得分':<8} {'稳定性':<8}")
        print("=" * 150)
        
        for _, row in cumulative_df.iterrows():
            transporter_list = row['transporter_list']
            if len(transporter_list) > 33:
                transporter_list = transporter_list[:30] + "..."
            
            print(f"{row['num_transporters']:<6} {transporter_list:<35} "
                  f"{row['total_capacity']:<10,.0f} {row['total_effective_capacity']:<10,.0f} "
                  f"{row['weighted_loss_rate']:<10.3f}% {row['effective_capacity_rate']:<8.1%} "
                  f"{row['weighted_comprehensive_score']:<8.1f} {row['weighted_stability_score']:<8.1f}")
        
        print("=" * 150)
        
        # 提供建议
        print(f"\\n运力配置建议:")
        print("-" * 60)
        
        # 基本需求 (20,304 m³/周)
        basic_need = 20304
        basic_transporters_needed = None
        
        # 1.5倍需求应对
        enhanced_need = basic_need * 1.5
        enhanced_transporters_needed = None
        
        # 2倍需求应对
        double_need = basic_need * 2
        double_transporters_needed = None
        
        for _, row in cumulative_df.iterrows():
            if basic_transporters_needed is None and row['total_effective_capacity'] >= basic_need:
                basic_transporters_needed = row['num_transporters']
            
            if enhanced_transporters_needed is None and row['total_effective_capacity'] >= enhanced_need:
                enhanced_transporters_needed = row['num_transporters']
            
            if double_transporters_needed is None and row['total_effective_capacity'] >= double_need:
                double_transporters_needed = row['num_transporters']
        
        print(f"满足基本需求({basic_need:,.0f} m³/周): 需要前 {basic_transporters_needed or '全部'} 家转运商")
        print(f"满足1.5倍需求({enhanced_need:,.0f} m³/周): 需要前 {enhanced_transporters_needed or '全部'} 家转运商")
        print(f"满足2倍需求({double_need:,.0f} m³/周): 需要前 {double_transporters_needed or '全部'} 家转运商")
        
        # 可视化累计分析
        self.visualize_cumulative_analysis(cumulative_df, sort_label)
        
        return {
            'individual_ranking': summary_df,
            'cumulative_analysis': cumulative_df,
            'sort_by': sort_by,
            'recommendations': {
                'basic_need': basic_need,
                'basic_transporters_needed': basic_transporters_needed,
                'enhanced_need': enhanced_need, 
                'enhanced_transporters_needed': enhanced_transporters_needed,
                'double_need': double_need,
                'double_transporters_needed': double_transporters_needed
            }
        }
    
    def visualize_cumulative_analysis(self, cumulative_df, sort_label):
        """可视化转运商累计分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'转运商累计运力分析 (排序: {sort_label})', fontsize=16, fontweight='bold')
        
        num_transporters = cumulative_df['num_transporters']
        
        # 1. 累计运力增长曲线
        ax1 = axes[0, 0]
        ax1.plot(num_transporters, cumulative_df['total_capacity'], 
                label='总运力', linewidth=3, color='blue', marker='o')
        ax1.plot(num_transporters, cumulative_df['total_effective_capacity'], 
                label='有效运力', linewidth=3, color='green', marker='s')
        
        # 添加需求线
        ax1.axhline(y=20304, color='red', linestyle='--', linewidth=2, label='基本需求 (20,304 m³)')
        ax1.axhline(y=30456, color='orange', linestyle='--', linewidth=2, label='1.5倍需求 (30,456 m³)')
        
        ax1.set_title('累计运力增长趋势')
        ax1.set_xlabel('转运商数量')
        ax1.set_ylabel('运力 (m³/周)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损耗率期望变化
        ax2 = axes[0, 1]
        ax2.plot(num_transporters, cumulative_df['weighted_loss_rate'], 
                linewidth=3, color='red', marker='o')
        
        for i, (x, y) in enumerate(zip(num_transporters, cumulative_df['weighted_loss_rate'])):
            if i % 2 == 0:  # 每隔一个点显示数值
                ax2.annotate(f'{y:.3f}%', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
        
        ax2.set_title('加权平均损耗率变化')
        ax2.set_xlabel('转运商数量')
        ax2.set_ylabel('损耗率 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 有效运力率变化
        ax3 = axes[1, 0]
        effective_rates = cumulative_df['effective_capacity_rate'] * 100
        ax3.plot(num_transporters, effective_rates, 
                linewidth=3, color='green', marker='s')
        
        for i, (x, y) in enumerate(zip(num_transporters, effective_rates)):
            if i % 2 == 0:
                ax3.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
        
        ax3.set_title('有效运力率变化')
        ax3.set_xlabel('转运商数量')
        ax3.set_ylabel('有效运力率 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 综合得分变化
        ax4 = axes[1, 1]
        ax4.plot(num_transporters, cumulative_df['weighted_comprehensive_score'], 
                linewidth=3, color='purple', marker='d', label='综合得分')
        ax4.plot(num_transporters, cumulative_df['weighted_stability_score'], 
                linewidth=3, color='orange', marker='^', label='稳定性得分')
        
        ax4.set_title('加权平均得分变化')
        ax4.set_xlabel('转运商数量')
        ax4.set_ylabel('得分')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'Pictures/cumulative_transporter_analysis_{sort_label.replace("（", "_").replace("）", "_").replace("、", "_")}.svg', 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_all_algorithms(self, material_type='C'):
        """比较所有优化算法的结果（转运商组合）"""
        self.logger.info("比较所有转运商组合优化算法的结果...")
        
        algorithms = ['min_loss', 'weighted_score', 'capacity_efficiency']
        comparison_results = {}
        
        for algorithm in algorithms:
            result = self.optimize_transporter_combination(material_type, algorithm)
            if result:
                comparison_results[algorithm] = result
        
        # 生成对比表格
        print("\\n转运商组合算法对比结果:")
        print("=" * 120)
        print(f"{'算法':<15} {'转运商数量':<10} {'转运商组合':<25} {'加权损耗率':<12} {'总损耗量(m³)':<15} {'运力利用率':<12}")
        print("-" * 120)
        
        for algorithm, result in comparison_results.items():
            combination = ', '.join(result['best_combination'])
            if len(combination) > 23:
                combination = combination[:20] + "..."
            
            total_loss = sum(plan['loss_volume'] for plan in result['allocation_plan'].values())
            
            print(f"{algorithm:<15} {result['combination_metrics']['num_transporters']:<10} "
                  f"{combination:<25} {result['combination_metrics']['weighted_avg_loss_rate']:<12.3f} "
                  f"{total_loss:<15.0f} {result['combination_metrics']['capacity_utilization']:<12.1%}")
        
        # 可视化对比结果
        self.visualize_combination_comparison(comparison_results)
        
        return comparison_results
    
    def visualize_combination_comparison(self, comparison_results):
        """可视化转运商组合对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('转运商组合选择算法对比分析', fontsize=16, fontweight='bold')
        
        algorithms = list(comparison_results.keys())
        algorithm_labels = {
            'min_loss': '最小损耗',
            'weighted_score': '加权评分', 
            'capacity_efficiency': '运力效率'
        }
        
        # 1. 损耗率对比
        ax1 = axes[0, 0]
        loss_rates = [comparison_results[alg]['combination_metrics']['weighted_avg_loss_rate'] 
                      for alg in algorithms]
        
        bars1 = ax1.bar([algorithm_labels[alg] for alg in algorithms], loss_rates, 
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, rate in zip(bars1, loss_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.3f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('加权平均损耗率对比')
        ax1.set_ylabel('损耗率 (%)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 转运商数量对比
        ax2 = axes[0, 1]
        num_transporters = [comparison_results[alg]['combination_metrics']['num_transporters'] 
                           for alg in algorithms]
        
        bars2 = ax2.bar([algorithm_labels[alg] for alg in algorithms], num_transporters,
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, num in zip(bars2, num_transporters):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{num}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('所需转运商数量对比')
        ax2.set_ylabel('转运商数量')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 运力利用率对比
        ax3 = axes[1, 0]
        utilization_rates = [comparison_results[alg]['combination_metrics']['capacity_utilization'] * 100
                           for alg in algorithms]
        
        bars3 = ax3.bar([algorithm_labels[alg] for alg in algorithms], utilization_rates,
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, rate in zip(bars3, utilization_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('运力利用率对比')
        ax3.set_ylabel('运力利用率 (%)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 总损耗量对比
        ax4 = axes[1, 1]
        total_losses = []
        for alg in algorithms:
            total_loss = sum(plan['loss_volume'] for plan in comparison_results[alg]['allocation_plan'].values())
            total_losses.append(total_loss)
        
        bars4 = ax4.bar([algorithm_labels[alg] for alg in algorithms], total_losses,
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, loss in zip(bars4, total_losses):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_losses)*0.01,
                    f'{loss:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('总损耗量对比')
        ax4.set_ylabel('损耗量 (m³)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Pictures/combination_algorithm_comparison.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_algorithm_comparison(self, comparison_results):
        """可视化算法对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('转运商选择算法对比分析', fontsize=16, fontweight='bold')
        
        algorithms = list(comparison_results.keys())
        algorithm_labels = {
            'min_loss': '最小损耗',
            'weighted_score': '加权评分', 
            'multi_objective': '多目标优化'
        }
        
        # 1. 损耗率对比
        ax1 = axes[0, 0]
        loss_rates = [comparison_results[alg]['best_transporter_data']['predicted_avg_loss_rate'] 
                      for alg in algorithms]
        
        bars1 = ax1.bar([algorithm_labels[alg] for alg in algorithms], loss_rates, 
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, rate in zip(bars1, loss_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.3f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('预测损耗率对比')
        ax1.set_ylabel('损耗率 (%)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 选择的转运商
        ax2 = axes[0, 1]
        selected_transporters = [comparison_results[alg]['best_transporter'] for alg in algorithms]
        
        # 统计转运商选择频次
        from collections import Counter
        transporter_counts = Counter(selected_transporters)
        
        transporters = list(transporter_counts.keys())
        counts = list(transporter_counts.values())
        
        bars2 = ax2.bar(transporters, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(transporters)], alpha=0.8)
        
        for bar, count in zip(bars2, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('算法选择的转运商')
        ax2.set_ylabel('选择次数')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 损耗成本对比
        ax3 = axes[1, 0]
        loss_costs = [comparison_results[alg]['estimated_loss_cost'] for alg in algorithms]
        
        bars3 = ax3.bar([algorithm_labels[alg] for alg in algorithms], loss_costs,
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, cost in zip(bars3, loss_costs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(loss_costs)*0.01,
                    f'{cost:,.0f}', ha='center', va='bottom', fontweight='bold', rotation=45)
        
        ax3.set_title('预计损耗成本对比')
        ax3.set_ylabel('损耗成本 (元)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 综合效率对比
        ax4 = axes[1, 1]
        efficiency_rates = [comparison_results[alg]['best_transporter_data']['effective_capacity_rate'] * 100
                           for alg in algorithms]
        
        bars4 = ax4.bar([algorithm_labels[alg] for alg in algorithms], efficiency_rates,
                       color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        for bar, rate in zip(bars4, efficiency_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('有效运输率对比')
        ax4.set_ylabel('有效运输率 (%)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Pictures/algorithm_comparison.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_results(self):
        """保存分析结果到Excel文件"""
        filename = f'DataFrames/转运商损耗率分析结果.xlsx'
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 1. 转运商基础分析
                if self.analysis_df is not None:
                    self.analysis_df.to_excel(writer, sheet_name='转运商基础分析', index=False)
                
                # 2. ARIMA模型参数
                if self.arima_models:
                    arima_summary = []
                    for name, model_info in self.arima_models.items():
                        arima_summary.append({
                            '转运商': name,
                            'ARIMA参数': str(model_info['best_params']),
                            'AIC': model_info['aic'],
                            'BIC': model_info['bic'],
                            'MAE': model_info['mae'],
                            'RMSE': model_info['rmse'],
                            '预测平均损耗率': np.mean(model_info['forecast'])
                        })
                    
                    arima_df = pd.DataFrame(arima_summary)
                    arima_df.to_excel(writer, sheet_name='ARIMA模型汇总', index=False)
                
                # 3. 预测结果
                if self.arima_models:
                    predictions_data = []
                    max_forecast_length = max(len(model_info['forecast']) for model_info in self.arima_models.values())
                    
                    for week in range(max_forecast_length):
                        week_data = {'周数': week + 1}
                        for name, model_info in self.arima_models.items():
                            if week < len(model_info['forecast']):
                                week_data[f'{name}_预测损耗率'] = model_info['forecast'][week]
                            else:
                                week_data[f'{name}_预测损耗率'] = np.nan
                        predictions_data.append(week_data)
                    
                    predictions_df = pd.DataFrame(predictions_data)
                    predictions_df.to_excel(writer, sheet_name='损耗率预测结果', index=False)
            
            self.logger.info(f"分析结果已保存至: {filename}")
            print(f"\\n分析结果已保存至: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"保存分析结果失败: {e}")
            return None


def main():
    """主函数 - 运行完整的转运商损耗率分析流程"""
    print("=" * 60)
    print("转运商损耗率分析与ARIMA预测系统")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = TransporterLossAnalyzer()
    
    # 1. 加载数据
    if not analyzer.load_data():
        print("数据加载失败，程序终止")
        return
    
    # 2. 分析转运商特征
    analysis_df = analyzer.analyze_transporter_characteristics()
    if analysis_df is None:
        print("转运商特征分析失败，程序终止")
        return
    
    # 3. 获取转运商排名
    ranking = analyzer.get_transporter_ranking('comprehensive_score')
    
    # 4. 可视化分析结果
    analyzer.visualize_transporter_analysis()
    
    # 5. 构建ARIMA预测模型
    successful_models = analyzer.build_all_arima_models(forecast_weeks=24)
    
    if successful_models == 0:
        print("ARIMA模型构建失败，无法进行预测分析")
        return
    
    # 6. 可视化部分转运商的ARIMA结果
    # 选择排名前3的转运商进行详细分析
    top_transporters = ranking.head(3)['transporter_name'].tolist()
    
    for transporter_name in top_transporters:
        if transporter_name in analyzer.arima_models:
            analyzer.visualize_arima_results(transporter_name)
    
    # 7. 转运商累计运力分析（新功能）
    print("\\n" + "=" * 60)
    print("转运商累计运力分析")
    print("=" * 60)
    
    # 按不同排序方式进行累计分析
    print("\\n1. 按预测损耗率排序的累计分析:")
    loss_rate_analysis = analyzer.analyze_cumulative_transporter_capacity('predicted_loss_rate')
    
    print("\\n2. 按综合得分排序的累计分析:")
    comprehensive_analysis = analyzer.analyze_cumulative_transporter_capacity('comprehensive_score')
    
    print("\\n3. 按稳定性得分排序的累计分析:")
    stability_analysis = analyzer.analyze_cumulative_transporter_capacity('stability_score')
    
    # 8. 优化转运商组合选择
    print("\\n" + "=" * 60)
    print("转运商组合选择优化分析")
    print("=" * 60)
    
    # 比较所有转运商组合算法
    combination_results = analyzer.compare_all_algorithms(material_type='C')
    
    # 9. 保存分析结果
    analyzer.save_analysis_results()
    
    print("\\n转运商损耗率分析完成！")
    
    return analyzer, combination_results, {
        'loss_rate_analysis': loss_rate_analysis,
        'comprehensive_analysis': comprehensive_analysis,
        'stability_analysis': stability_analysis
    }


if __name__ == "__main__":
    results = main()
    analyzer = results[0]
    combination_results = results[1] 
    cumulative_analyses = results[2]
