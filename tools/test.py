import argparse
import torch
from models.pointconv_interaction_networks import build_pointconv_interaction_nets, build_multi_frame_mlp
from tools.dataset import Dataset, collate_fn, move_to_gpu
from tools.runner import Model_Runner
from tools.utils import get_config, set_seed, seed_worker

torch.multiprocessing.set_sharing_strategy('file_system')

# config
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='a yaml file path')
parser.add_argument('--scenario', type=str, required=True, help='physion scenario')
parser.add_argument('--epoch', type=int, required=True, help='selected epoch for testing')
parser.add_argument('--seed', type=int, default=None, help='random seed')
args = parser.parse_args()
cfg = get_config(args)
phase = 'test'

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
datasets = {phase: Dataset(phase, cfg)}
dataloaders = {phase: 
    torch.utils.data.DataLoader(
        datasets[phase],
        batch_size=cfg[phase]['batch_sz'],
        shuffle=False,
        num_workers=cfg[phase]['num_workers'],
        collate_fn=collate_fn,
        worker_init_fn=worker_fn,
        generator=g
    )
}
test_generator = dataloaders[phase]

# setup model
pointconv_interaction_nets = build_pointconv_interaction_nets(cfg, phase)
multi_frame_mlp = build_multi_frame_mlp(cfg, phase)
criterion = torch.nn.L1Loss(reduction='none')
model_runner = Model_Runner([pointconv_interaction_nets, multi_frame_mlp], criterion, None, cfg)

# testing loop
for idx, batch in enumerate(test_generator):
    assert len(batch['seq_name']) == 1, "Processing only one scene at a time for testing."
    print("%.2f%% Completed, Processing scene: %s" % (float(idx + 1) / len(test_generator) * 100, batch['seq_name'][0]))
    with torch.no_grad():
        loss, _, _ = model_runner.compute_loss(move_to_gpu(batch, cfg['device']), phase=phase)