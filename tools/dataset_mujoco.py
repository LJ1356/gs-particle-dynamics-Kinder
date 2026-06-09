"""
MuJoCo sweep dataset adapter for the GS-Particle-Dynamics pipeline.

Produces the same output format as tools/dataset.py so that the paper's
Model_Runner, collate_fn, and training loop can be used unchanged.

Data source:
  /home/liang/claude/kindergarden/hdf5_data/sweep_canonical_10d_nocamera.hdf5

Geom inclusion (102 geoms, 51 000 points):
  kitchen_island  — static surface context
  robot_arm       — dynamic, action-driven
  swept_cube      — geom133-139, dynamic passive
  drawer          — geom53-55, dynamic passive

Per-point feature vector (161-dim = 59 + camera_num * max_object_num):
  unnorm_rotation  (4)   identity quaternion  [1,0,0,0]
  logit_opacity    (1)   0.0
  log_scale        (3)   0.0
  seg_color        (3)   0.0
  shs              (48)  0.0
  gs_soft_ids_vec  (102) one-hot from integer geom ID

Usage (mirrors tools/train.py):
  datasets = {phase: MuJoCoDataset(phase, cfg)}
  dataloaders = {phase: DataLoader(..., collate_fn=collate_fn)}
"""

import os
import copy
import numpy as np
import torch
import torch.utils.data

import h5py
from pytorch3d.structures import Pointclouds

from tools.dataset import (
    find_knns_across_levels,
    collate_fn,
    move_to_gpu,
)
from tools.utils import compute_velocity_input


HDF5_PATH = '/home/liang/claude/kindergarden/hdf5_data/sweep_canonical_10d_nocamera.hdf5'

# ──────────────────────────────────────────────
# Geom helpers (same logic as adapt_mujoco_hdf5.py)
# ──────────────────────────────────────────────

def classify_geom(name):
    """Return category string, or None to exclude."""
    if 'kitchen_island' in name:
        return 'kitchen_island'
    if any(x in name for x in ['kitchen_cooking_area', 'kitchen_left_corner', 'kitchen_left_side']):
        return None
    if name.startswith('geom'):
        num = int(name.split('_')[0][4:])
        if 133 <= num <= 139:
            return 'swept_cube'
        if num in [53, 54, 55]:
            return 'drawer'
        return None
    return 'robot_arm'


def reconstruct_positions(canonical_xyz, transforms):
    """
    canonical_xyz : (N, 3)
    transforms    : (T, 4, 4)
    returns       : (T, N, 3) world positions
    """
    R = transforms[:, :3, :3]
    t = transforms[:, :3, 3]
    return np.einsum('tij,nj->tni', R, canonical_xyz) + t[:, np.newaxis, :]


# ──────────────────────────────────────────────
# Feature vector builder
# ──────────────────────────────────────────────

def build_features(positions, obj_ids, camera_num, max_object_num):
    """
    Build the per-point feature vector expected by the paper's model.

    positions   : (N, 3)  world positions at this frame
    obj_ids     : (N,)    integer geom ID per point (0 to max_object_num-1)
    returns     : (N, 59 + camera_num * max_object_num)
    """
    N = positions.shape[0]

    unnorm_rot  = np.zeros((N, 4),  dtype=np.float32); unnorm_rot[:, 0] = 1.0  # [1,0,0,0]
    logit       = np.zeros((N, 1),  dtype=np.float32)
    log_scale   = np.zeros((N, 3),  dtype=np.float32)
    seg         = np.zeros((N, 3),  dtype=np.float32)
    shs         = np.zeros((N, 48), dtype=np.float32)

    # one-hot soft ID: (N, camera_num, max_object_num) → reshape to (N, camera_num * max_object_num)
    gs_soft_ids = np.zeros((N, camera_num, max_object_num), dtype=np.float32)
    for n in range(N):
        gs_soft_ids[n, 0, obj_ids[n]] = 1.0
    gs_soft_ids_vec = gs_soft_ids.reshape(N, camera_num * max_object_num)

    feats = np.concatenate([unnorm_rot, logit, log_scale, seg, shs, gs_soft_ids_vec], axis=1)
    return feats  # (N, 59 + camera_num * max_object_num)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class MuJoCoDataset(torch.utils.data.Dataset):

    def __init__(self, phase, cfg):
        self.cfg = cfg
        self.phase = phase
        self.lookahead_frames = (
            0 if phase == 'test'
            else cfg[phase]['seq_len'] - (cfg['model']['input_frame_num'] + 1)
        )
        assert self.lookahead_frames >= 0

        # frames needed per sample: 3 input (0,1,2) + 1 GT + lookahead
        self.frames_needed = 3 + 1 + self.lookahead_frames

        self.all_trials = []   # list of (positions (T,N,3), obj_ids (N,), demo_name)
        self.n_rollout  = 0
        self.mean_time_step = cfg[phase].get('frames_per_scene', 1) if phase == 'train' else 1

        self._load()

    # ── loading ────────────────────────────────

    def _load(self):
        hdf5_path = self.cfg['dataset']['data_dir']
        camera_num    = self.cfg['dataset']['camera_num']
        max_object_num = self.cfg['dataset']['max_object_num']

        with h5py.File(hdf5_path, 'r') as f:
            for demo_name in sorted(f['data'].keys()):
                demo = f[f'data/{demo_name}']
                positions, obj_ids = self._load_demo(demo, camera_num, max_object_num)
                T = positions.shape[0]
                if T < self.frames_needed:
                    print(f'{demo_name}: only {T} frames, need {self.frames_needed}, skipping.')
                    continue
                self.all_trials.append((positions, obj_ids, demo_name))
                self.n_rollout += 1

        assert self.n_rollout > 0, f'No valid demos found in {self.cfg["dataset"]["data_dir"]}'
        print(f'Loaded {self.n_rollout} demos, '
              f'N={self.all_trials[0][0].shape[1]} points, '
              f'lookahead={self.lookahead_frames}')

    def _load_demo(self, demo, camera_num, max_object_num):
        geom_keys = sorted(demo['geom_transforms'].keys())
        pts_per_geom = self.cfg['dataset'].get('pts_per_geom', 500)
        rng = np.random.default_rng(seed=42)  # fixed seed for reproducible subsampling
        all_pos, all_ids = [], []
        geom_id = 0
        for gk in geom_keys:
            if classify_geom(gk) is None:
                continue
            canonical  = demo[f'canonical_pointcloud/{gk}/xyz'][:]  # (500, 3)
            if pts_per_geom < canonical.shape[0]:
                idx = rng.choice(canonical.shape[0], pts_per_geom, replace=False)
                idx.sort()
                canonical = canonical[idx]
            transforms = demo[f'geom_transforms/{gk}'][:]           # (T, 4, 4)
            pos = reconstruct_positions(canonical, transforms)       # (T, pts, 3)
            all_pos.append(pos)
            all_ids.extend([geom_id] * canonical.shape[0])
            geom_id += 1
        positions = np.concatenate(all_pos, axis=1).astype(np.float32)  # (T, N, 3)
        obj_ids   = np.array(all_ids, dtype=np.int64)                   # (N,)
        return positions, obj_ids

    # ── length ────────────────────────────────

    def __len__(self):
        return self.n_rollout * self.mean_time_step

    # ── getitem ───────────────────────────────

    def __getitem__(self, idx):
        trial_idx = idx % self.n_rollout
        positions, obj_ids, demo_name = self.all_trials[trial_idx]
        T = positions.shape[0]

        # pick start frame
        max_start = T - self.frames_needed
        if self.phase == 'train':
            start = np.random.randint(0, max_start + 1)
        else:
            start = 0

        # window: frames [start, start+frames_needed)
        window = positions[start: start + self.frames_needed]  # (frames_needed, N, 3)
        # frame indices within window:
        #   0       → t-2 (for velocity of first input)
        #   1       → t-1 (first input frame  → seq_info[0])
        #   2       → t   (second input frame → seq_info[1], nn_idx comes from here)
        #   3       → GT frame (position[0])
        #   4, 5, … → lookahead frames (position[1], position[2], …)

        camera_num     = self.cfg['dataset']['camera_num']
        max_object_num = self.cfg['dataset']['max_object_num']
        scaling        = self.cfg['pointcloud']['scaling']

        example = {}
        example['seq_name']  = f'{demo_name}_t{start}'
        example['start_frame'] = start
        example['inv_rot']   = torch.eye(3, dtype=torch.float32)
        example['seq_info']  = {}
        example['gt_lookahead'] = []

        # ── build 2 input frames (ct=0 for frame idx 1, ct=1 for frame idx 2) ──
        for ct, frame_idx in enumerate([1, 2]):
            pos      = window[frame_idx]                               # (N, 3)
            pos_prev = window[frame_idx - 1]                           # (N, 3)

            # scale both position and velocity the same way (vel is a position difference)
            pos_scaled = pos * scaling
            vel_scaled = (pos - pos_prev) * scaling

            feats = build_features(pos_scaled, obj_ids, camera_num, max_object_num)  # (N, 161)

            data     = [pos_scaled, vel_scaled]
            data_aux = [feats]
            obj_seq_info, nn_idx = self._prepare_model_input(data, data_aux)

            example['seq_info'][ct] = obj_seq_info
            if ct == 1:
                example['nn_idx'] = nn_idx   # from most recent input frame

        # ── build GT + lookahead position tensors ──
        gt_positions = []
        gt_rotations = []
        for k in range(1 + self.lookahead_frames):
            frame_idx = 3 + k
            pos_k = window[frame_idx] * scaling                       # (N, 3) scaled
            gt_positions.append(torch.tensor(pos_k, dtype=torch.float32))
            # dummy identity quaternion rotation
            rot_k = np.zeros((pos_k.shape[0], 4), dtype=np.float32)
            rot_k[:, 0] = 1.0
            gt_rotations.append(torch.tensor(rot_k, dtype=torch.float32))

        gt = {
            'position': torch.stack(gt_positions),   # (1+lookahead, N, 3)
            'rotation': torch.stack(gt_rotations),   # (1+lookahead, N, 4)
        }
        example['gt'] = gt

        # lookahead: empty dicts (rendering loss disabled, only len() is checked)
        example['gt_lookahead'] = [{} for _ in range(self.lookahead_frames)]

        # test phase: save GT params for later visualization
        if self.phase == 'test':
            self._save_gt_params(example, window, obj_ids, scaling)

        return example

    # ── model input preparation (mirrors dataset.py:_prepare_model_input) ──

    def _prepare_model_input(self, data_curr, data_curr_aux):
        positions_curr, velocities_curr = data_curr
        assert len(data_curr_aux) == 1
        feats_curr = data_curr_aux[0]
        assert positions_curr.shape[0] == feats_curr.shape[0]
        assert positions_curr.shape[0] == velocities_curr.shape[0]

        levels = self.cfg['pointcloud']['downsampling_layer_num'] + 1
        idx_curr_list = {i: [] for i in range(levels)}
        pos_curr_list = {i: [] for i in range(levels)}
        feats_curr_list = {i: [] for i in range(levels)}
        nn_idx_self_list = {i: [] for i in range(levels)}
        nn_idx_forward_list = {i: [] for i in range(levels - 1)}
        nn_idx_propagate_list = {i: [] for i in range(levels - 1)}

        instance_idx = [0, positions_curr.shape[0]]

        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            pc_all, feat_all, grid_idx_all, nn_idx_all = find_knns_across_levels(
                positions_curr[st:ed],
                feats_curr[st:ed],
                self.cfg['pointcloud']['minimum_point_num'],
                self.cfg['pointcloud']['knn'],
                self.cfg['pointcloud']['knn_k_decay_factor'],
                self.cfg['pointcloud']['grid_size'],
                self.cfg['pointcloud']['scaling']
            )
            grid_idx_offset = st
            idx_curr_list[levels - 1].append(grid_idx_all[levels - 1] + grid_idx_offset)
            pos_curr_list[levels - 1].append(pc_all[levels - 1])
            feats_curr_list[levels - 1].append(feat_all[levels - 1])
            nn_idx_self_list[levels - 1].append(nn_idx_all['self'][levels - 1])
            for j in range(levels - 1):
                idx_curr_list[j].append(grid_idx_all[j] + grid_idx_offset)
                pos_curr_list[j].append(pc_all[j])
                feats_curr_list[j].append(feat_all[j])
                nn_idx_self_list[j].append(nn_idx_all['self'][j])
                nn_idx_forward_list[j].append(nn_idx_all['forward'][j])
                nn_idx_propagate_list[j].append(nn_idx_all['propagate'][j])

        grid_idx_curr = []
        pos_curr = []
        nn_idx_self = []
        nn_idx_forward = []
        nn_idx_propagate = []
        for j in range(levels - 1):
            pos_curr.append(Pointclouds(points=pos_curr_list[j], features=feats_curr_list[j]))
            nn_idx_self.append(Pointclouds(points=pos_curr_list[j], features=nn_idx_self_list[j]))
            nn_idx_forward.append(Pointclouds(points=pos_curr_list[j + 1], features=nn_idx_forward_list[j]))
            nn_idx_propagate.append(Pointclouds(points=pos_curr_list[j], features=nn_idx_propagate_list[j]))
            grid_idx_curr.append(torch.cat(idx_curr_list[j], dim=0))
        pos_curr.append(Pointclouds(points=pos_curr_list[levels - 1], features=feats_curr_list[levels - 1]))
        nn_idx_self.append(Pointclouds(points=pos_curr_list[levels - 1], features=nn_idx_self_list[levels - 1]))
        grid_idx_curr.append(torch.cat(idx_curr_list[levels - 1], dim=0))

        obj_seq_info = {}
        pos_packed = pos_curr[0].points_packed()
        vel_packed = torch.tensor(velocities_curr, dtype=torch.float32)
        assert vel_packed.shape[1] == 3
        obj_seq_info['pc']       = pos_curr
        obj_seq_info['grid_idx'] = grid_idx_curr
        obj_seq_info['velocity'] = compute_velocity_input(pos_packed, vel_packed, None, None, self.cfg)

        nn_idx = {'self': nn_idx_self, 'forward': nn_idx_forward, 'propagate': nn_idx_propagate}
        return obj_seq_info, nn_idx

    # ── test-phase GT save ────────────────────

    def _save_gt_params(self, example, window, obj_ids, scaling):
        output_dir = os.path.join(self.cfg['output_dir'], self.cfg['exp_name_epoch'])
        scene_dir  = os.path.join(output_dir, example['seq_name'])
        os.makedirs(scene_dir, exist_ok=True)
        params = {
            'means3D': window / scaling,   # unscaled positions (frames_needed, N, 3)
            'obj_ids': obj_ids,
        }
        np.savez(os.path.join(scene_dir, 'params_gt.npz'), **params)


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from tools.utils import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/config_mujoco.yaml')
    parser.add_argument('--scenario',    default='sweep')
    parser.add_argument('--seed',        default=None)
    args = parser.parse_args()

    cfg = get_config(args)

    print('Building train dataset...')
    ds = MuJoCoDataset('train', cfg)
    print(f'  len={len(ds)}')

    print('Fetching sample[0]...')
    sample = ds[0]

    print('  Keys:', list(sample.keys()))
    print('  seq_info frames:', list(sample['seq_info'].keys()))

    pc0 = sample['seq_info'][0]['pc'][0]
    print(f'  pc[0] points shape  : {pc0.points_packed().shape}')
    print(f'  pc[0] features shape: {pc0.features_packed().shape}')
    print(f'  velocity shape      : {sample["seq_info"][0]["velocity"].shape}')
    print(f'  gt position shape   : {sample["gt"]["position"].shape}')
    print(f'  gt_lookahead len    : {len(sample["gt_lookahead"])}')
    print(f'  nn_idx self levels  : {len(sample["nn_idx"]["self"])}')

    print('\nRunning collate_fn on 2 samples...')
    batch = collate_fn([ds[0], ds[1]])
    print('  batch keys:', list(batch.keys()))
    print('  batch pc[0] points:', batch['pc'][0][0].points_packed().shape)
    print('  batch gt len:', len(batch['gt']))

    print('\nSanity check passed.')
