#!/usr/bin/env python3
"""
可视化原始模型和改进模型的训练结果对比
比较三个数据集（BindingDB, BIOSNAP, DAVIS）的性能指标
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib

# 设置中文字体支持
# 尝试多个常见的中文字体
chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Microsoft YaHei', 
                 'SimHei', 'STSong', 'Arial Unicode MS']

# 获取系统可用字体
available_fonts = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# 选择第一个可用的中文字体
selected_font = None
for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font]
else:
    # 如果没有中文字体，使用默认字体并打印警告
    print("Warning: No Chinese font found. Chinese characters may not display correctly.")
    print(f"Available fonts: {list(available_fonts)[:10]}...")  # 显示前10个可用字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_results(result_path):
    """加载训练结果JSON文件"""
    try:
        with open(result_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found - {result_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON - {result_path}")
        return None

def extract_epoch_metrics(results):
    """从结果中提取每个epoch的指标"""
    if not results or 'epoch_history' not in results:
        return None
    
    epochs = []
    aurocs = []
    auprcs = []
    f1s = []
    losses = []
    
    for entry in results['epoch_history']:
        epochs.append(entry['epoch'])
        val = entry['validation']
        aurocs.append(val['auroc'])
        auprcs.append(val['auprc'])
        f1s.append(val['f1'])
        losses.append(val['loss'])
    
    return {
        'epochs': epochs,
        'auroc': aurocs,
        'auprc': auprcs,
        'f1': f1s,
        'loss': losses
    }

def plot_comparison_single_dataset(dataset_name, old_results, new_results, save_path=None):
    """
    为单个数据集绘制原始vs改进的4个指标对比图
    
    参数:
        dataset_name: 数据集名称
        old_results: 原始模型结果
        new_results: 改进模型结果
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} Dataset: Original vs Improved Model', fontsize=16, fontweight='bold')
    
    old_metrics = extract_epoch_metrics(old_results)
    new_metrics = extract_epoch_metrics(new_results)
    
    if old_metrics is None and new_metrics is None:
        print(f"Warning: No data available for {dataset_name}")
        plt.close(fig)
        return
    
    # 指标配置
    metrics_config = [
        ('auroc', 'AUROC', 'upper left', True),   # (key, title, legend_loc, higher_is_better)
        ('auprc', 'AUPRC', 'upper left', True),
        ('f1', 'F1 Score', 'upper left', True),
        ('loss', 'Loss', 'upper right', False)
    ]
    
    for idx, (metric_key, metric_title, legend_loc, higher_better) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]
        
        # 绘制原始模型
        if old_metrics:
            ax.plot(old_metrics['epochs'], old_metrics[metric_key], 
                   marker='o', linestyle='-', linewidth=2, markersize=4,
                   label='Original', color='#FF6B6B', alpha=0.8)
        
        # 绘制改进模型
        if new_metrics:
            ax.plot(new_metrics['epochs'], new_metrics[metric_key], 
                   marker='s', linestyle='-', linewidth=2, markersize=4,
                   label='Improved', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_title, fontsize=11)
        ax.set_title(f'{metric_title} Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc=legend_loc, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()

def plot_all_datasets_comparison(datasets_data, save_path=None):
    """
    绘制所有数据集的综合对比图
    每个指标一个子图，显示所有数据集的原始vs改进
    
    参数:
        datasets_data: {dataset_name: {'old': old_results, 'new': new_results}}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('All Datasets Performance: Original vs Improved Model', fontsize=18, fontweight='bold')
    
    metrics_config = [
        ('auroc', 'AUROC', 'upper left'),
        ('auprc', 'AUPRC', 'upper left'),
        ('f1', 'F1 Score', 'upper left'),
        ('loss', 'Loss', 'upper right')
    ]
    
    for idx, (metric_key, metric_title, legend_loc) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]
        
        bar_width = 0.35
        x_positions = np.arange(len(datasets_data))
        
        old_values = []
        new_values = []
        improvements = []
        dataset_names = []
        
        for dataset_name, data in datasets_data.items():
            dataset_names.append(dataset_name)
            
            # 获取最佳验证指标
            old_val = data['old']['best_validation'][metric_key] if data['old'] else 0
            new_val = data['new']['best_validation'][metric_key] if data['new'] else 0
            
            old_values.append(old_val)
            new_values.append(new_val)
            
            # 计算改进百分比
            if metric_key == 'loss':
                improvement = ((old_val - new_val) / old_val * 100) if old_val != 0 else 0
            else:
                improvement = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
            improvements.append(improvement)
        
        # 绘制柱状图 - 使用更强烈的对比色
        bars1 = ax.bar(x_positions - bar_width/2, old_values, bar_width, 
                       label='Original Model', alpha=0.85, color='#FF6B6B', 
                       edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_positions + bar_width/2, new_values, bar_width, 
                       label='Improved Model', alpha=0.85, color='#4ECDC4',
                       edgecolor='black', linewidth=1.5)
        
        # 在柱子上添加数值 - 更大更清晰
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 4),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 设置y轴范围，为标注留出适当空间
        y_max = max(max(old_values), max(new_values))
        y_min = min(min(old_values), min(new_values))
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.18)
        
        # 在两个柱子之间添加改进百分比标注
        for i, (x_pos, improvement) in enumerate(zip(x_positions, improvements)):
            y_pos = max(old_values[i], new_values[i]) + y_range * 0.08
            color = 'green' if improvement >= 0 else 'red'
            symbol = '↑' if improvement >= 0 else '↓'
            ax.text(x_pos, y_pos, f'{symbol} {abs(improvement):.2f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   color=color, bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='white', edgecolor=color, 
                                         linewidth=2, alpha=0.8))
        
        ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_title} Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(dataset_names, fontsize=12, fontweight='bold')
        ax.legend(loc=legend_loc, fontsize=11, framealpha=0.95, edgecolor='black', 
                 fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive comparison: {save_path}")
    
    plt.show()

def print_summary_table(datasets_data):
    """打印结果汇总表格"""
    print("\n" + "="*100)
    print("Performance Comparison Summary".center(100))
    print("="*100)
    
    header = f"{'Dataset':<12} {'Metric':<10} {'Original':<15} {'Improved':<15} {'Change':<15} {'Change%':<10}"
    print(header)
    print("-"*100)
    
    metrics = ['auroc', 'auprc', 'f1', 'loss']
    metric_names = {'auroc': 'AUROC', 'auprc': 'AUPRC', 'f1': 'F1', 'loss': 'Loss'}
    
    for dataset_name, data in datasets_data.items():
        for i, metric in enumerate(metrics):
            old_val = data['old']['best_validation'][metric] if data['old'] else 0
            new_val = data['new']['best_validation'][metric] if data['new'] else 0
            
            if metric == 'loss':
                change = old_val - new_val
                improvement = (change / old_val * 100) if old_val != 0 else 0
            else:
                change = new_val - old_val
                improvement = (change / old_val * 100) if old_val != 0 else 0
            
            change_str = f"{change:+.4f}"
            improvement_str = f"{improvement:+.2f}%"
            
            # 根据是否改进设置颜色标记
            if (metric != 'loss' and change > 0) or (metric == 'loss' and change > 0):
                marker = "✅"
            elif change < 0:
                marker = "❌"
            else:
                marker = "➖"
            
            dataset_col = dataset_name if i == 0 else ""
            print(f"{dataset_col:<12} {metric_names[metric]:<10} {old_val:<15.4f} {new_val:<15.4f} {change_str:<15} {improvement_str:<10} {marker}")
        
        print("-"*100)
    
    print("="*100)

def main():
    """主函数"""
    base_path = Path('./result')
    
    # 数据集列表
    datasets = ['BindingDB', 'BIOSNAP', 'DAVIS']
    
    # 加载所有数据
    datasets_data = {}
    
    for dataset in datasets:
        old_path = base_path / 'old' / dataset / 'training_results.json'
        new_path = base_path / 'new' / dataset / 'training_results.json'
        
        old_results = load_results(old_path)
        new_results = load_results(new_path)
        
        if old_results or new_results:
            datasets_data[dataset] = {
                'old': old_results,
                'new': new_results
            }
            
            # 为每个数据集绘制单独的详细对比图
            if old_results and new_results:
                save_path = f'comparison_{dataset.lower()}.png'
                plot_comparison_single_dataset(dataset, old_results, new_results, save_path)
    
    # 绘制所有数据集的综合对比
    if datasets_data:
        plot_all_datasets_comparison(datasets_data, 'comparison_all_datasets.png')
        
        # 打印汇总表格
        print_summary_table(datasets_data)
    else:
        print("Error: No training result files found!")
        print("Please ensure the following paths exist:")
        print("  - ./result/old/{BindingDB,BIOSNAP,DAVIS}/training_results.json")
        print("  - ./result/new/{BindingDB,BIOSNAP,DAVIS}/training_results.json")

if __name__ == '__main__':
    main()
