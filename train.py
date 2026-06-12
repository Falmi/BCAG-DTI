import copy
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis'],
                    default='', type=str, metavar='TASK',
                    help='Task name. Could be biosnap, bindingdb and davis.')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--warmup-epochs', default=3, type=int,
                    help='number of warmup epochs')
parser.add_argument('--weight-decay', default=1e-5, type=float,
                    help='weight decay for optimizer')


def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    loss_fct = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for stability
    
    with torch.no_grad():
        for i, (d, p, d_mask, p_mask, label) in enumerate(tqdm(data_generator, desc="Testing")):
            d = d.long().to(device)
            p = p.long().to(device)
            d_mask = d_mask.long().to(device)
            p_mask = p_mask.long().to(device)
            label = torch.from_numpy(np.array(label)).float().to(device)

            with autocast(enabled=use_cuda):
                logits_raw = torch.squeeze(model(d, p, d_mask, p_mask))  # raw logits
                loss = loss_fct(logits_raw, label)

            loss_accumulate += loss.item()
            count += 1

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits_raw).detach().cpu().numpy()
            label_ids = label.cpu().numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + probs.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr + 1e-8)

    f1 = 2 * precision * tpr / (tpr + precision + 1e-8)

    thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(thresholds) > 6 else 0.5

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1 if total1 > 0 else 0
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1] + 1e-8)
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1] + 1e-8)
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss


def main():
    args = parser.parse_args()
    config = BIN_config_DBPE(dataset=args.task)  # Pass dataset name for adaptive config
    config['batch_size'] = args.batch_size


    loss_history = []

    model = BIN_Interaction_Flat(**config)

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    # Dataset-specific learning rate adjustment - REVISED for better DAVIS performance
    dataset_lr_multiplier = {
        'davis': 1.2,      # DAVIS: slightly higher LR to escape local minima
        'biosnap': 1.0,
        'bindingdb': 1.0
    }
    adjusted_lr = args.lr * dataset_lr_multiplier.get(args.task.lower(), 1.0)
    
    # Dataset-specific weight decay - REVISED for better DAVIS performance
    dataset_wd_multiplier = {
        'davis': 0.5,      # DAVIS: lower weight decay to avoid underfitting
        'biosnap': 1.0,
        'bindingdb': 1.0
    }
    adjusted_wd = args.weight_decay * dataset_wd_multiplier.get(args.task.lower(), 1.0)
    
    print(f'Dataset: {args.task.upper()}')
    print(f'Adjusted LR: {adjusted_lr:.6f} (base: {args.lr:.6f})')
    print(f'Adjusted Weight Decay: {adjusted_wd:.6f} (base: {args.weight_decay:.6f})')
    
    # Use AdamW with dataset-specific weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=adjusted_wd)
    
    # Dataset-specific warmup epochs
    dataset_warmup = {
        'davis': 5,        # DAVIS: longer warmup for stable training
        'biosnap': 3,
        'bindingdb': 3
    }
    actual_warmup = dataset_warmup.get(args.task.lower(), args.warmup_epochs)
    print(f'Warmup epochs: {actual_warmup}')
    
    # Warmup + Cosine learning rate scheduler
    def warmup_cosine_schedule(epoch):
        if epoch < actual_warmup:
            return (epoch + 1) / actual_warmup
        else:
            progress = (epoch - actual_warmup) / max(1, args.epochs - actual_warmup)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_cosine_schedule)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=use_cuda)
    
    # Loss function with dataset-specific label smoothing
    # Label smoothing prevents overconfidence and improves generalization
    class LabelSmoothingBCEWithLogitsLoss(nn.Module):
        def __init__(self, smoothing=0.0):
            super().__init__()
            self.smoothing = smoothing
            self.bce = nn.BCEWithLogitsLoss()
        
        def forward(self, pred, target):
            if self.smoothing > 0:
                # Smooth the labels: 0 -> smoothing, 1 -> (1 - smoothing)
                target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return self.bce(pred, target)
    
    # Use label smoothing for DAVIS to reduce overfitting
    smoothing = 0.1 if args.task.lower() == 'davis' else 0.0
    loss_fct = LabelSmoothingBCEWithLogitsLoss(smoothing=smoothing)
    print(f'Using label smoothing: {smoothing}')
    
    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers,
              'drop_last': True,
              'pin_memory': True}  # Enable pin_memory for faster data transfer

    dataFolder = get_task(args.task)

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    # Dataset-specific early stopping patience
    dataset_patience = {
        'davis': 25,       # DAVIS: more patience for slow, stable convergence
        'biosnap': 15,
        'bindingdb': 15
    }
    patience = dataset_patience.get(args.task.lower(), 15)
    print(f'Early stopping patience: {patience}')
    
    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)
    best_epoch = 0
    best_val_metrics = {}
    epoch_history = []  # Store all epoch results
    patience_counter = 0

    with torch.set_grad_enabled(False):
        auc, auprc, f1, logits, loss = test(testing_generator, model_max)
        print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
            f1) + ' , Test loss: ' + str(loss))

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        for i, (d, p, d_mask, p_mask, label) in enumerate(tqdm(training_generator, desc=f"Epoch {epo + 1}/{args.epochs}")):
            d = d.long().to(device)
            p = p.long().to(device)
            d_mask = d_mask.long().to(device)
            p_mask = p_mask.long().to(device)
            label = torch.from_numpy(np.array(label)).float().to(device)

            # Mixed precision training
            with autocast(enabled=use_cuda):
                logits_raw = torch.squeeze(model(d, p, d_mask, p_mask))
                loss = loss_fct(logits_raw, label)

            loss_val = loss.item()
            loss_history.append(loss_val)
            epoch_loss += loss_val
            epoch_steps += 1

            opt.zero_grad()
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(opt)
            scaler.update()

            if (i % 1000 == 0):
                print(f'Training at Epoch {epo + 1} iteration {i} with loss {loss_val:.6f}')
        
        # Update learning rate per epoch (not per step)
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        current_lr = opt.param_groups[0]['lr']
        print(f'Epoch {epo + 1} - Avg Loss: {avg_epoch_loss:.6f}, LR: {current_lr:.6e}')
        
        # Step scheduler after each epoch
        scheduler.step()

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(validation_generator, model)
            
            # Record this epoch's results
            epoch_metrics = {
                'epoch': epo + 1,
                'validation': {
                    'auroc': float(auc),
                    'auprc': float(auprc),
                    'f1': float(f1),
                    'loss': float(loss)
                },
                'is_best': False,
                'learning_rate': float(current_lr)
            }
            
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
                best_epoch = epo + 1
                best_val_metrics = {
                    'epoch': best_epoch,
                    'auroc': auc,
                    'auprc': auprc,
                    'f1': f1,
                    'loss': loss
                }
                epoch_metrics['is_best'] = True
                patience_counter = 0
                
                # Save best model checkpoint
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model_max.state_dict() if not isinstance(model_max, nn.DataParallel) else model_max.module.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': best_val_metrics,
                }, f'best_model_checkpoint.pt')
                print(f'✓ New best model saved at epoch {epo + 1}!')
            else:
                patience_counter += 1
            
            epoch_history.append(epoch_metrics)
            
            print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                auprc) + ' , F1: ' + str(f1))
            
            # Early stopping check
            if patience_counter >= patience:
                print(f'\n⚠ Early stopping triggered at epoch {epo + 1} (patience: {patience})')
                break

    print('\n' + '='*80)
    print('BEST VALIDATION RESULTS')
    print('='*80)
    print(f"Best Epoch: {best_val_metrics['epoch']}")
    print(f"Best Validation AUROC: {best_val_metrics['auroc']:.4f}")
    print(f"Best Validation AUPRC: {best_val_metrics['auprc']:.4f}")
    print(f"Best Validation F1: {best_val_metrics['f1']:.4f}")
    print(f"Best Validation Loss: {best_val_metrics['loss']:.4f}")
    print('='*80 + '\n')

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            test_auc, test_auprc, test_f1, logits, test_loss = test(testing_generator, model_max)
            
            print('\n' + '='*80)
            print('FINAL TEST RESULTS (Best Model from Epoch {})'.format(best_epoch))
            print('='*80)
            print(f"Test AUROC: {test_auc:.4f}")
            print(f"Test AUPRC: {test_auprc:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print('='*80 + '\n')
            
    except Exception as e:
        print(f'Testing failed with error: {e}')
        test_auc = test_auprc = test_f1 = test_loss = 0
    
    return model_max, loss_history, best_val_metrics, test_auc, test_auprc, test_f1, test_loss, epoch_history


s = time()
model_max, loss_history, best_val_metrics, test_auc, test_auprc, test_f1, test_loss, epoch_history = main()
e = time()

print('\n' + '='*80)
print('TRAINING SUMMARY')
print('='*80)
print(f'Total Training Time: {e - s:.2f} seconds ({(e - s)/60:.2f} minutes)')
print(f'\nBest Validation Results (Epoch {best_val_metrics.get("epoch", "N/A")}):')
print(f'  - AUROC: {best_val_metrics.get("auroc", 0):.4f}')
print(f'  - AUPRC: {best_val_metrics.get("auprc", 0):.4f}')
print(f'  - F1: {best_val_metrics.get("f1", 0):.4f}')
print(f'  - Loss: {best_val_metrics.get("loss", 0):.4f}')

print(f'\nFinal Test Results:')
print(f'  - AUROC: {test_auc:.4f}')
print(f'  - AUPRC: {test_auprc:.4f}')
print(f'  - F1: {test_f1:.4f}')
print(f'  - Loss: {test_loss:.4f}')
print('='*80 + '\n')

# Plot loss history
lh = [loss.cpu().detach().numpy() if torch.is_tensor(loss) else loss for loss in loss_history]
lh = list(filter(lambda x: x < 1, lh))
plt.figure(figsize=(10, 6))
plt.plot(lh)
plt.title('Training Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_history.png')
print('Loss history plot saved to loss_history.png')
plt.close()

# Save full training results to JSON
import json
try:
    # get task name (safe to parse args again)
    task_name = parser.parse_args().task
except Exception:
    task_name = ''

results = {
    'task': task_name,
    'training_time_seconds': e - s,
    'best_validation': best_val_metrics,
    'test': {
        'auroc': float(test_auc),
        'auprc': float(test_auprc),
        'f1': float(test_f1),
        'loss': float(test_loss)
    },
    'epoch_history': epoch_history  # Add all epoch results
}

with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print('Training results saved to training_results.json')

# Also save epoch history separately for easier analysis
with open('epoch_history.json', 'w') as f:
    json.dump(epoch_history, f, indent=4)

print('Epoch history saved to epoch_history.json')
