"""
Training script for MuJoCo sweep dataset using the paper's PointConv model.

Mirrors tools/train.py but uses MuJoCoDataset instead of Dataset.

Usage (from project root):
  PYTHONPATH=. python tools/train_mujoco.py --config_path configs/config_mujoco.yaml
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

from models.pointconv_interaction_networks import build_pointconv_interaction_nets, build_multi_frame_mlp
from tools.runner import Model_Runner
from tools.dataset import move_to_gpu, collate_fn
from tools.dataset_mujoco import MuJoCoDataset
from tools.utils import get_config, set_seed, seed_worker

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--scenario',    type=str, default='sweep')
parser.add_argument('--seed',        type=int, default=None)
args = parser.parse_args()
cfg = get_config(args)

if cfg.get('exp_seed') is not None:
    set_seed(cfg['exp_seed'])
    worker_fn = seed_worker
    g = torch.Generator()
    g.manual_seed(cfg['exp_seed'])
else:
    worker_fn = None
    g = None

train_ds = MuJoCoDataset('train', cfg)
train_loader = DataLoader(
    train_ds,
    batch_size=cfg['train']['batch_sz'],
    shuffle=True,
    num_workers=cfg['train']['num_workers'],
    prefetch_factor=2 if cfg['train']['num_workers'] > 0 else None,
    collate_fn=collate_fn,
    drop_last=True,
    worker_init_fn=worker_fn,
    generator=g,
)

pointconv_interaction_nets = build_pointconv_interaction_nets(cfg, 'train')
multi_frame_mlp = build_multi_frame_mlp(cfg, 'train')

delta = 0.05 * cfg['pointcloud']['scaling']
criterion = torch.nn.HuberLoss(reduction='none', delta=delta)

all_params = [
    {'params': [p for n, p in pointconv_interaction_nets.named_parameters() if 'object_modeling' not in n],
     'weight_decay': 0.0, 'lr': cfg['train']['learning_rate']},
    {'params': [p for n, p in pointconv_interaction_nets.named_parameters() if 'object_modeling' in n],
     'weight_decay': 0.0, 'lr': cfg['train']['learning_rate']},
    {'params': list(multi_frame_mlp.parameters()),
     'weight_decay': 0.0, 'lr': cfg['train']['learning_rate']},
]

optimizer = torch.optim.AdamW(all_params)

warm_up_iters = cfg['train']['warm_up_epoch'] * len(train_loader)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warm_up_iters, total_iters=warm_up_iters),
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['train']['milestones']),
    ],
    milestones=[warm_up_iters],
)

model_runner = Model_Runner([pointconv_interaction_nets, multi_frame_mlp], criterion, optimizer, cfg)

for epoch in range(1, cfg['train']['epoch'] + 1):
    epoch_loss = []
    model_runner.put_models_in_train_mode()

    for idx, batch in enumerate(train_loader):
        loss, _, _ = model_runner.compute_loss(move_to_gpu(batch, cfg['device']), phase='train')
        model_runner.optimize(loss)
        epoch_loss.append(loss.item())

        if epoch <= cfg['train']['warm_up_epoch']:
            scheduler.step()

        pct = (idx + 1) / len(train_loader) * 100
        print(f'\rEpoch {epoch}/{cfg["train"]["epoch"]}  {pct:.1f}%  loss={loss.item():.6f}',
              end='', flush=True)

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    lr = optimizer.param_groups[0]['lr']
    print(f'\nEpoch {epoch}  avg_loss={avg_loss:.6f}  lr={lr:.2e}')

    if epoch > cfg['train']['warm_up_epoch']:
        scheduler.step()

    if epoch % 10 == 0 or epoch == cfg['train']['epoch']:
        model_runner.store_checkpoint(epoch)
        print(f'Checkpoint saved at epoch {epoch}.')
