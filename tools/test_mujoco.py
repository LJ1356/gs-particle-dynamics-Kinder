"""
Evaluation script for the MuJoCo sweep dataset.

Mirrors tools/test.py: it swaps Dataset -> MuJoCoDataset and lets the paper's
Model_Runner perform the full-trajectory autoregressive rollout (3 input frames,
predict every remaining frame to the end of the clip). The rollout writes
params.npz (predictions) per scene and the dataset writes params_gt.npz (ground
truth) per scene, both in original (metre) scale.

After the rollout this script reads those two files back and reports:
  - Delta metric: fraction of points within 5 / 10 / 20 cm of GT (final frame)
  - Chamfer distance: predicted vs GT point cloud (final frame)
  - Per-group breakdown (cube / drawer / wiper / robot / static) via parent_ids

avg_delta and chamfer_dist are reused from tools/compute_metrics.py.
compute_metrics.visualize() is skipped — it needs 4DGS reconstructions we don't have.

Usage (from project root):
  PYTHONPATH=. python tools/test_mujoco.py --config_path configs/config_mujoco.yaml --epoch 50
"""

import os
import argparse

import numpy as np
import torch

from models.pointconv_interaction_networks import build_pointconv_interaction_nets, build_multi_frame_mlp
from tools.dataset import collate_fn, move_to_gpu
from tools.dataset_mujoco import MuJoCoDataset, PARENT_TO_ID
from tools.runner import Model_Runner
from tools.compute_metrics import avg_delta, chamfer_dist
from tools.utils import get_config, set_seed, seed_worker

torch.multiprocessing.set_sharing_strategy('file_system')

THRESHOLDS = [0.05, 0.1, 0.2]  # metres
ID_TO_PARENT = {v: k for k, v in PARENT_TO_ID.items()}


def run_rollout(cfg, phase='test'):
    """Run the autoregressive rollout for every test scene, saving params.npz/params_gt.npz."""
    if cfg['exp_seed'] is not None:
        set_seed(cfg['exp_seed'])
        worker_fn = seed_worker
        g = torch.Generator()
        g.manual_seed(cfg['exp_seed'])
    else:
        worker_fn = None
        g = None

    dataset = MuJoCoDataset(phase, cfg)

    # The rollout length is bounded by cfg[phase]['seq_len'] (runner loops over
    # range(seq_len - 1)). Set it from the longest test demo so the full
    # trajectory is always rolled out, regardless of how long future test clips are.
    max_frames = max(trial[0].shape[0] for trial in dataset.all_trials)
    required = max_frames + 1  # need seq_len >= T - 1 to reach the final break; +1 is a safe margin
    if cfg[phase]['seq_len'] < required:
        print(f'Raising {phase} seq_len {cfg[phase]["seq_len"]} -> {required} '
              f'to cover the longest demo ({max_frames} frames).')
        cfg[phase]['seq_len'] = required

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg[phase]['batch_sz'],
        shuffle=False,
        num_workers=cfg[phase]['num_workers'],
        collate_fn=collate_fn,
        worker_init_fn=worker_fn,
        generator=g,
    )

    pointconv_interaction_nets = build_pointconv_interaction_nets(cfg, phase)
    multi_frame_mlp = build_multi_frame_mlp(cfg, phase)
    criterion = torch.nn.L1Loss(reduction='none')
    model_runner = Model_Runner([pointconv_interaction_nets, multi_frame_mlp], criterion, None, cfg)

    scene_names = []
    for idx, batch in enumerate(loader):
        assert len(batch['seq_name']) == 1, 'Processing only one scene at a time for testing.'
        seq_name = batch['seq_name'][0]
        scene_names.append(seq_name)
        print('%.1f%% rollout, scene: %s' % (float(idx + 1) / len(loader) * 100, seq_name))
        with torch.no_grad():
            model_runner.compute_loss(move_to_gpu(batch, cfg['device']), phase=phase)

    return scene_names


# ──────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────

def _delta_counts(pred_xyz, gt_xyz):
    """Return (counts_per_threshold, num_points). counts[i] = #points within THRESHOLDS[i]."""
    if pred_xyz.shape[0] == 0:
        return [0.0] * len(THRESHOLDS), 0
    counts, n = avg_delta(pred_xyz, gt_xyz, THRESHOLDS)
    return counts, n


def evaluate_scene(scene_dir):
    """Compute final-frame delta/chamfer for one scene, overall and per parent group.

    Returns a dict mapping group name ('overall' or a parent name) ->
        {'counts': [c5, c10, c20], 'n': int, 'chamfer': float}.
    """
    pred = np.load(os.path.join(scene_dir, 'params.npz'))
    gt = np.load(os.path.join(scene_dir, 'params_gt.npz'))

    pred_traj = torch.tensor(pred['means3D'], dtype=torch.float32)   # (P, N, 3)
    gt_full = torch.tensor(gt['means3D'], dtype=torch.float32)       # (T, N, 3)
    parent_ids = gt['parent_ids']                                   # (N,)

    # rollout predicts frames 3 .. T-1; align pred[i] <-> gt_full[3 + i]
    num_pred = pred_traj.shape[0]
    gt_traj = gt_full[3:3 + num_pred]
    assert gt_traj.shape == pred_traj.shape, f'{gt_traj.shape} vs {pred_traj.shape}'

    pred_final = pred_traj[-1]   # (N, 3)
    gt_final = gt_traj[-1]       # (N, 3)

    results = {}
    counts, n = _delta_counts(pred_final, gt_final)
    results['overall'] = {
        'counts': counts, 'n': n,
        'chamfer': chamfer_dist(pred_final, gt_final).item(),
    }

    for pid in np.unique(parent_ids):
        mask = parent_ids == pid
        p_sub = pred_final[mask]
        g_sub = gt_final[mask]
        counts, n = _delta_counts(p_sub, g_sub)
        results[ID_TO_PARENT.get(int(pid), f'parent_{pid}')] = {
            'counts': counts, 'n': n,
            'chamfer': chamfer_dist(p_sub, g_sub).item(),
        }
    return results


def aggregate_and_report(scene_names, cfg):
    output_dir = os.path.join(cfg['output_dir'], cfg['exp_name_epoch'])

    # accumulators: group -> [sum_c5, sum_c10, sum_c20], total_n, [chamfers]
    sum_counts = {}
    sum_n = {}
    chamfers = {}

    for seq_name in scene_names:
        scene_dir = os.path.join(output_dir, seq_name)
        res = evaluate_scene(scene_dir)
        for group, r in res.items():
            if group not in sum_counts:
                sum_counts[group] = [0.0] * len(THRESHOLDS)
                sum_n[group] = 0
                chamfers[group] = []
            for i in range(len(THRESHOLDS)):
                sum_counts[group][i] += r['counts'][i]
            sum_n[group] += r['n']
            chamfers[group].append(r['chamfer'])

    # ordered report: overall first, then parent groups in canonical order
    ordered = ['overall'] + [p for p in PARENT_TO_ID if p in sum_counts]

    header = (f'{"group":<16} {"npts":>7} '
              f'{"d<5cm":>8} {"d<10cm":>8} {"d<20cm":>8} {"chamfer":>9}')
    lines = ['', f'Final-frame metrics over {len(scene_names)} scene(s)  '
                 f'[{cfg["exp_name_epoch"]}]', header, '-' * len(header)]
    for group in ordered:
        n = sum_n[group] + 1e-8
        fracs = [sum_counts[group][i] / n for i in range(len(THRESHOLDS))]
        cham = float(np.mean(chamfers[group]))
        lines.append(f'{group:<16} {sum_n[group]:>7d} '
                     f'{fracs[0]:>8.4f} {fracs[1]:>8.4f} {fracs[2]:>8.4f} {cham:>9.4f}')
    report = '\n'.join(lines)
    print(report)

    out_path = os.path.join(output_dir, 'mujoco_metrics.txt')
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'\nMetrics written to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='a yaml file path')
    parser.add_argument('--scenario', type=str, default='sweep', help='scenario name')
    parser.add_argument('--epoch', type=int, required=True, help='checkpoint epoch to evaluate')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    cfg = get_config(args)

    scene_names = run_rollout(cfg, phase='test')
    aggregate_and_report(scene_names, cfg)
