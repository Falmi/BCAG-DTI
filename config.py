def BIN_config_DBPE(dataset='default'):
    config = {}
    config['batch_size'] = 16
    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['train_epoch'] = 13
    config['max_drug_seq'] = 50
    config['max_protein_seq'] = 545
    config['emb_size'] = 384
    
    # Dataset-specific dropout rates
    if dataset.lower() == 'davis':
        config['dropout_rate'] = 0.1  # DAVIS: lower dropout to preserve learned features
    elif dataset.lower() == 'biosnap':
        config['dropout_rate'] = 0.15
    else:
        config['dropout_rate'] = 0.15
    
    #DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3
    
    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 12
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 78192
    
    # Advanced features
    config['use_cross_attention'] = True
    config['use_multi_scale_pooling'] = True
    # Disable attention pooling for DAVIS to reduce complexity
    config['use_attention_pooling'] = False if dataset.lower() == 'davis' else True
    config['use_moe'] = False  # Mixture of Experts (experimental)
    
    return config