"""
快速测试脚本 - 验证模型改进
"""
import torch
import numpy as np
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat

def test_model_improvements():
    print("="*80)
    print("测试 MolTrans 模型改进")
    print("="*80)
    
    # 配置
    config = BIN_config_DBPE()
    print(f"\n✓ 配置加载成功")
    print(f"  - Embedding size: {config['emb_size']}")
    print(f"  - Dropout rate: {config['dropout_rate']}")
    print(f"  - Flat dim: {config['flat_dim']}")
    
    # 创建模型
    try:
        model = BIN_Interaction_Flat(**config)
        print(f"\n✓ 模型创建成功")
        
        # 检查新组件
        assert hasattr(model, 'cross_attention'), "❌ Missing cross_attention module"
        print(f"  ✓ Cross-Attention 模块存在")
        
        assert hasattr(model, 'global_pool_avg'), "❌ Missing global_pool_avg"
        assert hasattr(model, 'global_pool_max'), "❌ Missing global_pool_max"
        print(f"  ✓ Multi-scale Pooling 模块存在")
        
        # 检查分类器
        decoder_layers = list(model.decoder.children())
        print(f"  ✓ Decoder 包含 {len(decoder_layers)} 层")
        
        # 检查是否使用 LayerNorm 和 GELU
        has_layernorm = any(isinstance(layer, torch.nn.LayerNorm) for layer in decoder_layers)
        has_gelu = any(isinstance(layer, torch.nn.GELU) for layer in decoder_layers)
        
        if has_layernorm:
            print(f"  ✓ 使用 LayerNorm")
        else:
            print(f"  ⚠ 未检测到 LayerNorm")
            
        if has_gelu:
            print(f"  ✓ 使用 GELU 激活函数")
        else:
            print(f"  ⚠ 未检测到 GELU")
        
    except Exception as e:
        print(f"\n❌ 模型创建失败: {e}")
        return False
    
    # 测试前向传播
    try:
        print(f"\n测试前向传播...")
        batch_size = 4
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 创建dummy输入
        d = torch.randint(0, config['input_dim_drug'], (batch_size, config['max_drug_seq'])).to(device)
        p = torch.randint(0, config['input_dim_target'], (batch_size, config['max_protein_seq'])).to(device)
        d_mask = torch.ones(batch_size, config['max_drug_seq']).to(device)
        p_mask = torch.ones(batch_size, config['max_protein_seq']).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(d, p, d_mask, p_mask)
        
        print(f"  ✓ 前向传播成功")
        print(f"  - 输入 batch size: {batch_size}")
        print(f"  - 输出 shape: {output.shape}")
        print(f"  - 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        assert output.shape == (batch_size, 1), f"输出shape错误: {output.shape}"
        print(f"  ✓ 输出 shape 正确 (raw logits)")
        
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 统计参数量
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 模型大小: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
    except Exception as e:
        print(f"\n⚠ 参数统计失败: {e}")
    
    # 测试BCEWithLogitsLoss
    try:
        print(f"\n测试 BCEWithLogitsLoss...")
        loss_fct = torch.nn.BCEWithLogitsLoss()
        labels = torch.randint(0, 2, (batch_size,)).float().to(device)
        loss = loss_fct(output.squeeze(), labels)
        print(f"  ✓ BCEWithLogitsLoss 工作正常")
        print(f"  - Loss value: {loss.item():.4f}")
        
    except Exception as e:
        print(f"\n❌ Loss 计算失败: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！模型改进已成功应用")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_model_improvements()
    exit(0 if success else 1)
