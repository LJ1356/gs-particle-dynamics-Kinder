"""
MuJoCo sweep dataset adapter for the GS-Particle-Dynamics pipeline.

Produces the same output format as tools/dataset.py so that the paper's
Model_Runner, collate_fn, and training loop can be used unchanged.

Data source:
  /home/liang/claude/kindergarden/hdf5_data/sweep_canonical_10d_nocamera.hdf5

Included objects (63 geoms, 3 150 points @ pts_per_geom=50):
  static_tabletop  — kitchen_island_shelf3_geom7 (surface cubes slide on)
  drawer           — kitchen_island_drawer_s1c1_* + geom53/54/55
  wiper            — geom133 + geom134 (one tool, two geoms)
  cube_0..4        — geom135..139 (passive rigid objects)
  arm_link_0..7    — Kinova Gen3 arm links
  gripper_palm     — Robotiq 2F-85 base/body/arm_plate
  right_finger     — right finger mechanism
  left_finger      — left finger mechanism

Per-point feature vector (59 + camera_num × max_object_num dims):
  unnorm_rotation  (4)   identity quaternion [1,0,0,0]
  logit_opacity    (1)   0.0
  log_scale        (3)   0.0
  seg_color        (3)   0.0
  shs              (48)  0.0
  gs_soft_ids_vec  (K)   one-hot from selected object_id_mode

object_id_mode options (set in config_mujoco.yaml):
  "geom"   — one ID per included geom (63 IDs)
  "part"   — physical part level (19 IDs)  ← default
  "parent" — group level (9 IDs)
"""

import os
import re
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.utils.data
from pytorch3d.structures import Pointclouds

from tools.dataset import find_knns_across_levels, collate_fn, move_to_gpu
from tools.utils import compute_velocity_input


# ──────────────────────────────────────────────────────────────
# Geom number helper
# ──────────────────────────────────────────────────────────────

def _geom_num(name):
    """Extract trailing geom number, e.g. 'driver_geom162' → 162. Returns -1 if absent."""
    m = re.search(r'geom(\d+)$', name)
    return int(m.group(1)) if m else -1


# ──────────────────────────────────────────────────────────────
# Inclusion filter
# ──────────────────────────────────────────────────────────────

def classify_geom(name):
    """Return category string or None to exclude.

    Excluded (Phase 1 decision + Phase 2 refinement):
      - kitchen_cooking_area, kitchen_left_corner, kitchen_left_side
      - all kitchen_island geoms except shelf3 and s1c1 drawer (static_body excluded)
      - all other numbered geoms not in the explicit include list
    """
    if any(x in name for x in ['kitchen_cooking_area', 'kitchen_left_corner', 'kitchen_left_side']):
        return None
    if 'kitchen_island' in name:
        if name == 'kitchen_island_shelf3_geom7':
            return 'static_tabletop'
        if 's1c1' in name:
            return 'drawer'
        return None  # static_body excluded
    if name.startswith('geom'):
        try:
            num = int(name.split('_')[0][4:])
        except (ValueError, IndexError):
            return None
        if num in (53, 54, 55):   return 'drawer'
        if num in (133, 134):     return 'wiper'
        if 135 <= num <= 139:     return 'swept_cube'
        return None
    return 'robot_arm'


# ──────────────────────────────────────────────────────────────
# Hierarchical label assignment
# ──────────────────────────────────────────────────────────────

_PART_ORDER = [
    'static_tabletop',
    'drawer', 'wiper',
    'cube_0', 'cube_1', 'cube_2', 'cube_3', 'cube_4',
    'arm_link_0', 'arm_link_1', 'arm_link_2', 'arm_link_3',
    'arm_link_4', 'arm_link_5', 'arm_link_6', 'arm_link_7',
    'gripper_palm', 'right_finger', 'left_finger',
]
_PARENT_ORDER = [
    'static_context',
    'drawer', 'wiper',
    'cube_0', 'cube_1', 'cube_2', 'cube_3', 'cube_4',
    'robot',
]

PART_TO_ID   = {p: i for i, p in enumerate(_PART_ORDER)}
PARENT_TO_ID = {p: i for i, p in enumerate(_PARENT_ORDER)}
NUM_PARTS    = len(PART_TO_ID)    # 19
NUM_PARENTS  = len(PARENT_TO_ID)  # 9


def _match_part(name):
    """Return (part_name, parent_name, category, constraint) or None if unrecognised."""
    num = _geom_num(name)

    if name == 'kitchen_island_shelf3_geom7':
        return ('static_tabletop', 'static_context', 'static_context', 'static')

    if 's1c1' in name or num in (53, 54, 55):
        return ('drawer', 'drawer', 'passive_articulated', 'x_axis_only')

    if num in (133, 134):
        return ('wiper', 'wiper', 'passive_object', 'free_rigid')

    _cube_map = {135: 'cube_0', 136: 'cube_1', 137: 'cube_2', 138: 'cube_3', 139: 'cube_4'}
    if num in _cube_map:
        cn = _cube_map[num]
        return (cn, cn, 'passive_object', 'free_rigid')

    _arm_links = [
        ('gen3__base_link',           'arm_link_0'),
        ('shoulder_link',             'arm_link_1'),
        ('half_arm_1_link',           'arm_link_2'),
        ('half_arm_2_link',           'arm_link_3'),
        ('forearm_link',              'arm_link_4'),
        ('spherical_wrist_1_link',    'arm_link_5'),
        ('spherical_wrist_2_link',    'arm_link_6'),
        ('bracelet_with_vision_link', 'arm_link_7'),
    ]
    for substr, part_name in _arm_links:
        if substr in name:
            return (part_name, 'robot', 'robot_context', 'action_driven')

    if num in (140, 141, 142, 143, 160, 161):
        return ('gripper_palm', 'robot', 'robot_context', 'action_driven')

    if 'robot_right' in name or (162 <= num <= 173):
        return ('right_finger', 'robot', 'robot_context', 'action_driven')

    if 'robot_left' in name or (174 <= num <= 185):
        return ('left_finger', 'robot', 'robot_context', 'action_driven')

    return None


def assign_hierarchical_labels(included_geom_names):
    """Return (labels_dict, unassigned_list).

    labels_dict maps geom_name → {
        part_id, part_name, parent_id, parent_name, category, constraint
    }
    """
    labels, unassigned = {}, []
    for name in included_geom_names:
        result = _match_part(name)
        if result is None:
            unassigned.append(name)
        else:
            part_name, parent_name, category, constraint = result
            labels[name] = {
                'part_id':     PART_TO_ID[part_name],
                'part_name':   part_name,
                'parent_id':   PARENT_TO_ID[parent_name],
                'parent_name': parent_name,
                'category':    category,
                'constraint':  constraint,
            }
    return labels, unassigned


# ──────────────────────────────────────────────────────────────
# Feature vector builder
# ──────────────────────────────────────────────────────────────

def build_features(positions, obj_ids, camera_num, max_object_num):
    """Build the per-point feature vector expected by the paper's model.

    positions   : (N, 3)  scaled world positions
    obj_ids     : (N,)    integer IDs in [0, max_object_num)
    returns     : (N, 59 + camera_num * max_object_num)
    """
    N = positions.shape[0]
    unnorm_rot = np.zeros((N, 4),  dtype=np.float32); unnorm_rot[:, 0] = 1.0
    logit      = np.zeros((N, 1),  dtype=np.float32)
    log_scale  = np.zeros((N, 3),  dtype=np.float32)
    seg        = np.zeros((N, 3),  dtype=np.float32)
    shs        = np.zeros((N, 48), dtype=np.float32)

    gs_soft_ids = np.zeros((N, camera_num, max_object_num), dtype=np.float32)
    gs_soft_ids[np.arange(N), 0, obj_ids] = 1.0
    gs_soft_ids_vec = gs_soft_ids.reshape(N, camera_num * max_object_num)

    return np.concatenate([unnorm_rot, logit, log_scale, seg, shs, gs_soft_ids_vec], axis=1)


# ──────────────────────────────────────────────────────────────
# Position reconstruction
# ──────────────────────────────────────────────────────────────

def reconstruct_positions(canonical_xyz, transforms):
    """canonical_xyz (N,3), transforms (T,4,4) → world positions (T,N,3)."""
    R = transforms[:, :3, :3]
    t = transforms[:, :3, 3]
    return np.einsum('tij,nj->tni', R, canonical_xyz) + t[:, np.newaxis, :]


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class MuJoCoDataset(torch.utils.data.Dataset):

    def __init__(self, phase, cfg):
        self.cfg   = cfg
        self.phase = phase
        self.lookahead_frames = (
            0 if phase == 'test'
            else cfg[phase]['seq_len'] - (cfg['model']['input_frame_num'] + 1)
        )
        assert self.lookahead_frames >= 0
        self.frames_needed  = 3 + 1 + self.lookahead_frames
        self.all_trials     = []
        self.n_rollout      = 0
        self.mean_time_step = cfg[phase].get('frames_per_scene', 1) if phase == 'train' else 1
        self._load()

    # ── loading ───────────────────────────────────────────────

    def _data_path(self):
        """Pick the HDF5 file for the current phase (train and test use separate files).

        Falls back to a single `data_dir` if a per-phase key isn't set.
        """
        key = 'train' if self.phase in ('train', 'train_hem') else self.phase
        ds = self.cfg['dataset']
        path = ds.get(f'{key}_data_dir', ds.get('data_dir'))
        assert path is not None, (
            f"No data path for phase '{self.phase}': set dataset.{key}_data_dir "
            f"(or dataset.data_dir) in the config."
        )
        return path

    def _load(self):
        first_labels = None
        self.data_path = self._data_path()
        with h5py.File(self.data_path, 'r') as f:
            for demo_name in sorted(f['data'].keys()):
                demo = f[f'data/{demo_name}']
                positions, geom_ids, part_ids, parent_ids, labels = self._load_demo(demo)
                if first_labels is None:
                    first_labels = labels
                T = positions.shape[0]
                if T < self.frames_needed:
                    print(f'{demo_name}: {T} frames < {self.frames_needed} needed, skipping.')
                    continue
                self.all_trials.append((positions, geom_ids, part_ids, parent_ids, demo_name))
                self.n_rollout += 1

        assert self.n_rollout > 0, f'No valid demos in {self.data_path}'
        self._print_grouping_summary(first_labels)
        self._validate_config(first_labels)

    def _load_demo(self, demo):
        geom_keys    = sorted(demo['geom_transforms'].keys())
        pts_per_geom = self.cfg['dataset'].get('pts_per_geom', 500)
        rng = np.random.default_rng(seed=42)

        included = [gk for gk in geom_keys if classify_geom(gk) is not None]
        labels, unassigned = assign_hierarchical_labels(included)
        if unassigned:
            print(f'  WARNING: {len(unassigned)} unassigned geoms: {unassigned}')

        all_pos, all_geom_ids, all_part_ids, all_parent_ids = [], [], [], []
        for geom_idx, gk in enumerate(included):
            canonical = demo[f'canonical_pointcloud/{gk}/xyz'][:]
            if pts_per_geom < canonical.shape[0]:
                idx = rng.choice(canonical.shape[0], pts_per_geom, replace=False)
                idx.sort()
                canonical = canonical[idx]
            transforms = demo[f'geom_transforms/{gk}'][:]
            n = canonical.shape[0]
            lbl = labels[gk]

            all_pos.append(reconstruct_positions(canonical, transforms))
            all_geom_ids.extend([geom_idx]        * n)
            all_part_ids.extend([lbl['part_id']]   * n)
            all_parent_ids.extend([lbl['parent_id']] * n)

        positions  = np.concatenate(all_pos, axis=1).astype(np.float32)
        geom_ids   = np.array(all_geom_ids,   dtype=np.int64)
        part_ids   = np.array(all_part_ids,   dtype=np.int64)
        parent_ids = np.array(all_parent_ids, dtype=np.int64)
        return positions, geom_ids, part_ids, parent_ids, labels

    def _print_grouping_summary(self, labels):
        mode         = self.cfg['dataset'].get('object_id_mode', 'part')
        pts_per_geom = self.cfg['dataset'].get('pts_per_geom', 500)

        part_geom_count   = defaultdict(int)
        parent_geom_count = defaultdict(int)
        for lbl in labels.values():
            part_geom_count[lbl['part_name']]   += 1
            parent_geom_count[lbl['parent_name']] += 1

        print(f'\n{"─"*62}')
        print(f'MuJoCoDataset  |  {self.n_rollout} demo(s)  |  mode: {mode}  |  '
              f'{pts_per_geom} pts/geom')
        print(f'Included geoms: {len(labels)}  '
              f'| Part IDs: {NUM_PARTS}  '
              f'| Parent IDs: {NUM_PARENTS}  '
              f'| Total pts/frame: {len(labels) * pts_per_geom}')
        print()

        for parent_name in _PARENT_ORDER:
            if parent_name not in parent_geom_count:
                continue
            pg = parent_geom_count[parent_name]
            print(f'  [{parent_name}]  {pg} geoms  {pg * pts_per_geom} pts')
            seen = set()
            for lbl in labels.values():
                pn = lbl['part_name']
                if lbl['parent_name'] == parent_name and pn not in seen:
                    seen.add(pn)
                    g = part_geom_count[pn]
                    print(f'    part_id={PART_TO_ID[pn]:2d}  {pn:20s}  '
                          f'{g} geom(s)  {g*pts_per_geom} pts  [{lbl["constraint"]}]')
        print(f'{"─"*62}\n')

    def _validate_config(self, labels):
        mode    = self.cfg['dataset'].get('object_id_mode', 'part')
        cfg_max = self.cfg['dataset']['max_object_num']
        expected = {'geom': len(labels), 'part': NUM_PARTS, 'parent': NUM_PARENTS}[mode]
        if cfg_max != expected:
            print(f'WARNING: config max_object_num={cfg_max} but {mode} mode '
                  f'expects {expected}. Update configs/config_mujoco.yaml.')
        feat_dim = 59 + self.cfg['dataset']['camera_num'] * cfg_max
        print(f'Feature dim: 59 + {self.cfg["dataset"]["camera_num"]} × {cfg_max} = {feat_dim}')

    # ── length / getitem ──────────────────────────────────────

    def __len__(self):
        return self.n_rollout * self.mean_time_step

    def __getitem__(self, idx):
        trial_idx = idx % self.n_rollout
        positions, geom_ids, part_ids, parent_ids, demo_name = self.all_trials[trial_idx]
        T = positions.shape[0]

        if self.phase == 'test':
            # full-trajectory autoregressive rollout: 3 input frames, predict frames 3..T-1
            start = 0
            window = positions
            num_gt = T - 3
        else:
            max_start = T - self.frames_needed
            start = np.random.randint(0, max_start + 1)
            window = positions[start: start + self.frames_needed]
            num_gt = 1 + self.lookahead_frames

        camera_num    = self.cfg['dataset']['camera_num']
        max_object_num = self.cfg['dataset']['max_object_num']
        scaling       = self.cfg['pointcloud']['scaling']
        mode          = self.cfg['dataset'].get('object_id_mode', 'part')
        obj_ids       = {'geom': geom_ids, 'part': part_ids, 'parent': parent_ids}[mode]

        example = {
            'seq_name':   f'{demo_name}_t{start}',
            'start_frame': start,
            'inv_rot':    torch.eye(3, dtype=torch.float32),
            'seq_info':   {},
        }

        for ct, frame_idx in enumerate([1, 2]):
            pos_scaled = window[frame_idx] * scaling
            vel_scaled = (window[frame_idx] - window[frame_idx - 1]) * scaling
            feats = build_features(pos_scaled, obj_ids, camera_num, max_object_num)
            obj_seq_info, nn_idx = self._prepare_model_input([pos_scaled, vel_scaled], [feats])
            example['seq_info'][ct] = obj_seq_info
            if ct == 1:
                example['nn_idx'] = nn_idx

        gt_positions, gt_rotations = [], []
        for k in range(num_gt):
            pos_k = window[3 + k] * scaling
            gt_positions.append(torch.tensor(pos_k, dtype=torch.float32))
            rot_k = np.zeros((pos_k.shape[0], 4), dtype=np.float32); rot_k[:, 0] = 1.0
            gt_rotations.append(torch.tensor(rot_k, dtype=torch.float32))

        example['gt'] = {
            'position': torch.stack(gt_positions),
            'rotation': torch.stack(gt_rotations),
        }
        example['gt_lookahead'] = [{} for _ in range(self.lookahead_frames)]

        # metadata for debugging / per-category evaluation
        example['geom_ids']   = torch.from_numpy(geom_ids)
        example['part_ids']   = torch.from_numpy(part_ids)
        example['parent_ids'] = torch.from_numpy(parent_ids)

        if self.phase == 'test':
            self._save_gt_params(example, window, geom_ids, part_ids, parent_ids, scaling)

        return example

    # ── model input preparation ───────────────────────────────

    def _prepare_model_input(self, data_curr, data_curr_aux):
        positions_curr, velocities_curr = data_curr
        feats_curr = data_curr_aux[0]
        assert positions_curr.shape[0] == feats_curr.shape[0] == velocities_curr.shape[0]

        levels = self.cfg['pointcloud']['downsampling_layer_num'] + 1
        idx_curr_list    = {i: [] for i in range(levels)}
        pos_curr_list    = {i: [] for i in range(levels)}
        feats_curr_list  = {i: [] for i in range(levels)}
        nn_idx_self_list = {i: [] for i in range(levels)}
        nn_idx_forward_list   = {i: [] for i in range(levels - 1)}
        nn_idx_propagate_list = {i: [] for i in range(levels - 1)}

        st, ed = 0, positions_curr.shape[0]
        pc_all, feat_all, grid_idx_all, nn_idx_all = find_knns_across_levels(
            positions_curr[st:ed],
            feats_curr[st:ed],
            self.cfg['pointcloud']['minimum_point_num'],
            self.cfg['pointcloud']['knn'],
            self.cfg['pointcloud']['knn_k_decay_factor'],
            self.cfg['pointcloud']['grid_size'],
            self.cfg['pointcloud']['scaling'],
        )
        idx_curr_list[levels - 1].append(grid_idx_all[levels - 1] + st)
        pos_curr_list[levels - 1].append(pc_all[levels - 1])
        feats_curr_list[levels - 1].append(feat_all[levels - 1])
        nn_idx_self_list[levels - 1].append(nn_idx_all['self'][levels - 1])
        for j in range(levels - 1):
            idx_curr_list[j].append(grid_idx_all[j] + st)
            pos_curr_list[j].append(pc_all[j])
            feats_curr_list[j].append(feat_all[j])
            nn_idx_self_list[j].append(nn_idx_all['self'][j])
            nn_idx_forward_list[j].append(nn_idx_all['forward'][j])
            nn_idx_propagate_list[j].append(nn_idx_all['propagate'][j])

        pos_curr, nn_idx_self, nn_idx_forward, nn_idx_propagate, grid_idx_curr = [], [], [], [], []
        for j in range(levels - 1):
            pos_curr.append(Pointclouds(points=pos_curr_list[j], features=feats_curr_list[j]))
            nn_idx_self.append(Pointclouds(points=pos_curr_list[j], features=nn_idx_self_list[j]))
            nn_idx_forward.append(Pointclouds(points=pos_curr_list[j + 1], features=nn_idx_forward_list[j]))
            nn_idx_propagate.append(Pointclouds(points=pos_curr_list[j], features=nn_idx_propagate_list[j]))
            grid_idx_curr.append(torch.cat(idx_curr_list[j], dim=0))
        pos_curr.append(Pointclouds(points=pos_curr_list[levels - 1], features=feats_curr_list[levels - 1]))
        nn_idx_self.append(Pointclouds(points=pos_curr_list[levels - 1], features=nn_idx_self_list[levels - 1]))
        grid_idx_curr.append(torch.cat(idx_curr_list[levels - 1], dim=0))

        vel_packed = torch.tensor(velocities_curr, dtype=torch.float32)
        assert vel_packed.shape[1] == 3
        obj_seq_info = {
            'pc':       pos_curr,
            'grid_idx': grid_idx_curr,
            'velocity': compute_velocity_input(pos_curr[0].points_packed(), vel_packed, None, None, self.cfg),
        }
        return obj_seq_info, {'self': nn_idx_self, 'forward': nn_idx_forward, 'propagate': nn_idx_propagate}

    # ── test-phase GT save ────────────────────────────────────

    def _save_gt_params(self, example, window, geom_ids, part_ids, parent_ids, scaling):
        # window is in original (metre) scale; predictions are saved at the same scale
        # by runner._update_pred_log_using_gs_format (points * 1/scaling), so store as-is.
        output_dir = os.path.join(self.cfg['output_dir'], self.cfg['exp_name_epoch'])
        scene_dir  = os.path.join(output_dir, example['seq_name'])
        os.makedirs(scene_dir, exist_ok=True)
        np.savez(os.path.join(scene_dir, 'params_gt.npz'),
                 means3D=window,
                 geom_ids=geom_ids,
                 part_ids=part_ids,
                 parent_ids=parent_ids)


# ──────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from tools.utils import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/config_mujoco.yaml')
    parser.add_argument('--scenario',    default='sweep')
    parser.add_argument('--seed',        default=None)
    args = parser.parse_args()
    cfg  = get_config(args)

    ds = MuJoCoDataset('train', cfg)
    print(f'Dataset length: {len(ds)}')

    sample = ds[0]
    pc     = sample['seq_info'][1]['pc'][0]
    print(f'Points/scene    : {pc.points_packed().shape[0]}')
    print(f'Feature dim     : {pc.features_packed().shape[1]}')
    print(f'Velocity shape  : {sample["seq_info"][0]["velocity"].shape}')
    print(f'GT position     : {sample["gt"]["position"].shape}')
    print(f'Part IDs unique : {sample["part_ids"].unique().tolist()}')
    print(f'Parent IDs uniq : {sample["parent_ids"].unique().tolist()}')

    batch = collate_fn([ds[0], ds[1]])
    print(f'Batch pc[0]     : {batch["pc"][0][0].points_packed().shape}')
    print('Sanity check passed.')
