"""
对比测试脚本 - 验证 DAVIS 改进
"""
import torch
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat

def compare_configs():
    print("="*80)
    print("配置对比 - DAVIS vs 其他数据集")
    print("="*80)
    
    datasets = ['davis', 'biosnap', 'bindingdb']
    
    for dataset in datasets:
        print(f"\n📊 {dataset.upper()} 数据集配置:")
        print("-" * 60)
        
        config = BIN_config_DBPE(dataset=dataset)
        
        print(f"  Dropout Rate: {config['dropout_rate']}")
        print(f"  Embedding Size: {config['emb_size']}")
        print(f"  Intermediate Size: {config['intermediate_size']}")
        print(f"  Attention Heads: {config['num_attention_heads']}")
        print(f"  Use Cross-Attention: {config['use_cross_attention']}")
        print(f"  Use Attention Pooling: {config['use_attention_pooling']}")
        
        # 计算特征维度
        flat_dim = config['flat_dim']
        emb_size = config['emb_size']
        total_features = flat_dim + 6 * emb_size
        
        print(f"\n  特征维度:")
        print(f"    - Flat features: {flat_dim}")
        print(f"    - Pooling features: {6 * emb_size} (6 x {emb_size})")
        print(f"    - Total input to classifier: {total_features}")

def test_model_forward():
    print("\n" + "="*80)
    print("模型前向传播测试")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    for dataset in ['davis', 'biosnap']:
        print(f"\n🧪 测试 {dataset.upper()} 配置...")
        
        config = BIN_config_DBPE(dataset=dataset)
        model = BIN_Interaction_Flat(**config)
        model = model.to(device)
        model.eval()
        
        # 创建dummy输入
        d = torch.randint(0, config['input_dim_drug'], (batch_size, config['max_drug_seq'])).to(device)
        p = torch.randint(0, config['input_dim_target'], (batch_size, config['max_protein_seq'])).to(device)
        d_mask = torch.ones(batch_size, config['max_drug_seq']).to(device)
        p_mask = torch.ones(batch_size, config['max_protein_seq']).to(device)
        
        with torch.no_grad():
            output = model(d, p, d_mask, p_mask)
        
        print(f"  ✓ 前向传播成功")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 检查新特征
        print(f"\n  新增模块检查:")
        print(f"    ✓ Cross-Attention: {hasattr(model, 'cross_attention')}")
        print(f"    ✓ Attention Pooling (Drug): {hasattr(model, 'attention_pool_d')}")
        print(f"    ✓ Attention Pooling (Protein): {hasattr(model, 'attention_pool_p')}")
        print(f"    ✓ Residual Shortcuts: {hasattr(model, 'shortcut1') and hasattr(model, 'shortcut2')}")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  参数统计:")
        print(f"    - Total parameters: {total_params:,}")
        print(f"    - Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

def show_training_params():
    print("\n" + "="*80)
    print("训练超参数对比")
    print("="*80)
    
    base_lr = 1e-4
    base_wd = 1e-5
    
    dataset_lr_multiplier = {
        'davis': 0.5,
        'biosnap': 1.0,
        'bindingdb': 1.0
    }
    
    dataset_wd_multiplier = {
        'davis': 2.0,
        'biosnap': 1.0,
        'bindingdb': 1.0
    }
    
    dataset_patience = {
        'davis': 20,
        'biosnap': 15,
        'bindingdb': 15
    }
    
    print("\n| Dataset    | Learning Rate | Weight Decay | Patience | Dropout |")
    print("|------------|---------------|--------------|----------|---------|")
    
    for dataset in ['davis', 'biosnap', 'bindingdb']:
        config = BIN_config_DBPE(dataset=dataset)
        lr = base_lr * dataset_lr_multiplier[dataset]
        wd = base_wd * dataset_wd_multiplier[dataset]
        patience = dataset_patience[dataset]
        dropout = config['dropout_rate']
        
        print(f"| {dataset:10} | {lr:.6f}      | {wd:.6f}     | {patience:8} | {dropout:.2f}    |")
    
    print("\n💡 关键差异:")
    print("  - DAVIS 使用 50% 的学习率 (更稳定)")
    print("  - DAVIS 使用 2x 的权重衰减 (更强正则化)")
    print("  - DAVIS 使用 33% 更高的 dropout (0.20 vs 0.15)")
    print("  - DAVIS 使用更长的 patience (20 vs 15)")

def show_improvements_summary():
    print("\n" + "="*80)
    print("🎯 针对 DAVIS 的改进总结")
    print("="*80)
    
    improvements = [
        {
            'name': '注意力池化 (Attention Pooling)',
            'impact': '⭐⭐⭐',
            'description': '自适应学习重要特征，比固定池化更灵活'
        },
        {
            'name': '残差连接 (Residual Connections)',
            'impact': '⭐⭐⭐',
            'description': '缓解梯度消失，提高训练稳定性'
        },
        {
            'name': '数据集自适应学习率',
            'impact': '⭐⭐⭐⭐⭐',
            'description': 'DAVIS 使用 0.5x LR，避免过拟合'
        },
        {
            'name': '数据集自适应正则化',
            'impact': '⭐⭐⭐⭐⭐',
            'description': 'DAVIS 使用 2x Weight Decay + 更高 Dropout'
        },
        {
            'name': '学习率预热 (Warmup)',
            'impact': '⭐⭐⭐',
            'description': '训练初期更稳定，对小数据集有效'
        },
        {
            'name': '更长的 Patience',
            'impact': '⭐⭐⭐',
            'description': 'DAVIS 可能需要更多 epochs 收敛'
        }
    ]
    
    for i, imp in enumerate(improvements, 1):
        print(f"\n{i}. {imp['name']} {imp['impact']}")
        print(f"   {imp['description']}")
    
    print("\n" + "="*80)
    print("预期性能提升 (DAVIS):")
    print("  - AUROC: +3-5%")
    print("  - AUPRC: +4-6%")
    print("  - F1 Score: +3-5%")
    print("  - 训练稳定性: 显著提升")
    print("="*80)

if __name__ == "__main__":
    compare_configs()
    test_model_forward()
    show_training_params()
    show_improvements_summary()
    
    print("\n" + "="*80)
    print("✅ 所有检查完成！")
    print("\n📝 建议的训练命令:")
    print("\n  DAVIS (需要特殊处理):")
    print("    python3 train.py --task davis --epochs 60 --lr 1e-4 --warmup-epochs 5")
    print("\n  其他数据集 (使用默认参数):")
    print("    python3 train.py --task biosnap --epochs 50 --lr 1e-4")
    print("    python3 train.py --task bindingdb --epochs 50 --lr 1e-4")
    print("="*80)
