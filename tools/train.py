import os
import time
import argparse
import torch
from models.pointconv_interaction_networks import build_pointconv_interaction_nets, build_multi_frame_mlp
from tools.runner import Model_Runner
from tools.dataset import Dataset, move_to_gpu, collate_fn
from tools.utils import get_config, set_seed, seed_worker

torch.multiprocessing.set_sharing_strategy('file_system')

# config
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='a yaml file path')
parser.add_argument('--scenario', type=str, required=True, help='physion scenario')
parser.add_argument('--seed', type=int, default=None, help='random seed')
args = parser.parse_args()
cfg = get_config(args)

# fix random seeds if specified
if cfg['exp_seed'] is not None:
    set_seed(cfg['exp_seed'])
    worker_fn = seed_worker
    g = torch.Generator()
    g.manual_seed(cfg['exp_seed'])
else:
    worker_fn = None
    g = None

# generate data
hard_examples = {}
hard_examples_frames = {}
datasets = {phase: 
    Dataset(phase, cfg, hard_examples, hard_examples_frames) for phase in ['train', 'train_hem']
}

if cfg['train']['hard_example_mining']:
    dataloaders = {x:
        torch.utils.data.DataLoader(
            datasets[x],
            batch_size=cfg[x]['batch_sz'],
            shuffle=True if x == 'train' else False,
            num_workers=cfg[x]['num_workers'],
            prefetch_factor=4 if cfg[x]['num_workers'] > 0 else None,
            collate_fn=collate_fn,
            drop_last=True if x == 'train' else False,
            worker_init_fn=worker_fn,
            generator=g
        ) for x in ['train', 'train_hem']
    }
    train_generator = dataloaders['train']
    train_hem_generator = dataloaders['train_hem']
else:
    dataloaders = {x:
        torch.utils.data.DataLoader(
            datasets[x],
            batch_size=cfg[x]['batch_sz'],
            shuffle=True if x == 'train' else False,
            num_workers=cfg[x]['num_workers'],
            prefetch_factor=4 if cfg[x]['num_workers'] > 0 else None,
            collate_fn=collate_fn,
            drop_last=True if x == 'train' else False,
            worker_init_fn=worker_fn,
            generator=g
        ) for x in ['train']
    }
    train_generator = dataloaders['train']

# setup model
pointconv_interaction_nets = build_pointconv_interaction_nets(cfg, 'train')
multi_frame_mlp = build_multi_frame_mlp(cfg, 'train')

delta = 0.05 * cfg['pointcloud']['scaling']
criterion = torch.nn.HuberLoss(reduction='none', delta=delta)

obj_pconv_params = [p for name, p in pointconv_interaction_nets.named_parameters() if "object_modeling" not in name]
int_pconv_params = [p for name, p in pointconv_interaction_nets.named_parameters() if "object_modeling" in name]
multi_frame_mlp_params = list(multi_frame_mlp.parameters())

# total_params = sum(p.numel() for p in obj_pconv_params + int_pconv_params + multi_frame_mlp_params)
# print(f"Total number of parameters: {total_params}")

# fixed lr settings for different parts of the model for now
all_params = [
    {'params': obj_pconv_params, 'weight_decay': 0.0, 'lr': cfg['train']['learning_rate']}, 
    {'params': int_pconv_params, 'weight_decay': 0.0, 'lr': cfg['train']['learning_rate']},
    {'params': multi_frame_mlp_params, 'weight_decay': 0.0, 'lr': cfg['train']['learning_rate']},
]

warm_up_iter_per_epoch = len(train_generator)
warm_up_milestone = cfg['train']['warm_up_epoch'] * warm_up_iter_per_epoch

# setup optimizer
optimizer = torch.optim.AdamW(all_params)

scheduler1 = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1/warm_up_milestone, total_iters=warm_up_milestone
)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=cfg['train']['milestones']
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler1, scheduler2], milestones=[warm_up_milestone]
)
model_runner = Model_Runner([pointconv_interaction_nets, multi_frame_mlp], criterion, optimizer, cfg)

for epoch_num in range(1, cfg['train']['epoch'] + 1):
    # training loop
    epoch_loss = []
    model_runner.put_models_in_train_mode()
    for idx, batch in enumerate(train_generator):
        loss, _, _ = model_runner.compute_loss(move_to_gpu(batch, cfg['device']), phase='train')
        model_runner.optimize(loss)
        epoch_loss.append(loss.data.item())        
        # print(loss.data.item())        
        if epoch_num <= cfg['train']['warm_up_epoch']:
            scheduler.step()
        curr_p = float(idx + 1) / len(train_generator) * 100
        print("\rCurrent Epoch Training %.2f%% Completed. Current Epoch: %d/%d" % (curr_p, epoch_num, cfg['train']['epoch']), end='', flush=True)

    print("Average Epoch Loss: ", sum(epoch_loss) / len(epoch_loss), "Current LR: ", optimizer.param_groups[0]['lr'])

    if epoch_num > cfg['train']['warm_up_epoch']:
        scheduler.step()

    if epoch_num == cfg['train']['epoch']:
        model_runner.store_checkpoint(epoch_num)

    if not cfg['train']['hard_example_mining']:
        continue

    # time this portion
    start_time = time.time()
    # hard example mining 
    train_point_loss = 0.0
    train_frame_loss = 0.0
    total_point_num = 0
    total_frame_num = 0
    hard_examples = {}
    hard_examples_frames = {}
    model_runner.put_models_in_eval_mode()
    for idx, batch in enumerate(train_hem_generator):
        with torch.no_grad():
            _, point_loss, frame_loss = model_runner.compute_loss(move_to_gpu(batch, cfg['device']), phase='train_hem')
        train_point_loss += point_loss.sum().item()
        total_point_num += point_loss.shape[0]
        train_frame_loss += frame_loss.sum().item()
        total_frame_num += frame_loss.shape[0]

        if cfg['train']['hard_example_mining']:
            # hard example mining
            point_sz_per_batch = [t.item() for t in batch['batch_idx'][0]]
            point_loss_split = torch.split(point_loss, point_sz_per_batch, dim=0)        
            point_loss_avg_per_example = [pl.mean().item() for pl in point_loss_split]
            frame_loss_per_example = [frame_loss[fl_idx].item() for fl_idx in range(frame_loss.shape[0])]
            loss_per_example = [pa + fa for pa, fa in zip(point_loss_avg_per_example, frame_loss_per_example)]

            # loss_max_per_example = [pl.max().item() for pl in point_loss_split]                
            for loss_avg, seq_name, start_frame in zip(loss_per_example, batch['seq_name'], batch['start_frame']):
            # for loss_max, seq_name, start_frame in zip(loss_max_per_example, batch['seq_name'], batch['start_frame']):
                if seq_name not in hard_examples: hard_examples[seq_name] = []
                if seq_name not in hard_examples_frames: hard_examples_frames[seq_name] = []
                hard_examples[seq_name].append(loss_avg)
                # hard_examples[seq_name].append(loss_max)
                hard_examples_frames[seq_name].append(start_frame)
        
        # track hem progress and print it out
        curr_p = float(idx + 1) / len(train_hem_generator) * 100
        print("\rCurrent Epoch Hard Example Mining %.2f%% Completed." % (curr_p), end='', flush=True)

    hem_time = time.time() - start_time
    print("Hard Example Mining Time for Epoch %d: %.2f seconds" % (epoch_num, hem_time))

    # store train hem loss
    epoch_pos_loss = train_point_loss / total_point_num
    epoch_frame_loss = train_frame_loss / total_frame_num
    with open(os.path.join(cfg['exp_dir'], cfg['exp_name'], 'train_hem_loss.txt'), 'a') as f:
        f.write('Epoch %d, Loss: %.6f\n' % (epoch_num, epoch_pos_loss + epoch_frame_loss))
    with open(os.path.join(cfg['exp_dir'], cfg['exp_name'], 'train_hem_point_loss.txt'), 'a') as f:
        f.write('Epoch %d, Point Loss: %.6f\n' % (epoch_num, epoch_pos_loss))
    with open(os.path.join(cfg['exp_dir'], cfg['exp_name'], 'train_hem_frame_loss.txt'), 'a') as f:
        f.write('Epoch %d, Frame Loss: %.6f\n' % (epoch_num, epoch_frame_loss))
    
    # update hard examples in the dataset
    datasets['train'].update_hard_examples(hard_examples, hard_examples_frames)
    # reset the train data loader
    dataloaders['train'] = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=cfg['train']['batch_sz'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        prefetch_factor=4,
        collate_fn=collate_fn,
        drop_last=True
    )
    train_generator = dataloaders['train']