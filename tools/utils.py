import torch
import yaml
import os
import random
import numpy as np
import torch.nn.functional as F


def get_config(args):

    with open(args.config_path, 'rb') as yaml_config:
        cfg = yaml.safe_load(yaml_config)
    cfg['epoch_sel'] = args.epoch if hasattr(args, 'epoch') else None
    cfg['dataset']['scenario'] = args.scenario
    cfg['pointcloud']['downsampling_layer_num'] = len(cfg['pointcloud']['grid_size'])
    cfg['pointcloud']['minimum_point_num'] = cfg['pointcloud']['knn'] # set a heuristic minimum point count to keep KNN search valid.
    cfg['output_dir'] = os.path.join(cfg['output_dir'], args.scenario)
    # for additional data loader used when hard_example_mining is enabled
    cfg['train_hem']['batch_sz'] = cfg['train']['batch_sz']
    cfg['train_hem']['view_num'] = cfg['train']['view_num']
    cfg['train_hem']['num_workers'] = cfg['train']['num_workers']
    cfg['train_hem']['seq_len'] = cfg['train']['seq_len']

    cfg['exp_seed'] = args.seed
    if cfg['exp_seed'] is not None:
        cfg['exp_name_epoch'] = cfg['exp_name'] + f"_seed_{args.seed}" + f"_epoch{cfg['epoch_sel']}" + f"_{args.scenario}" # used only during testing
        cfg['exp_name'] = cfg['exp_name'] + f"_seed_{args.seed}" + f"_{args.scenario}"
    else:
        cfg['exp_name_epoch'] = cfg['exp_name'] + f"_epoch{cfg['epoch_sel']}" + f"_{args.scenario}" # used only during testing
        cfg['exp_name'] = cfg['exp_name'] + f"_{args.scenario}"

    return cfg


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_velocity_input(next_pc, pos_vel, rot_vel, scale, config):

    grav_coords = next_pc[:, [config['dataset']['gravity_axis']]]
    grav_coords = torch.clamp(grav_coords, max=1.0 * config['pointcloud']['scaling'])
    # velocity = torch.cat((pos_vel, grav_coords, rot_vel, scale), dim=1)
    # velocity = torch.cat((pos_vel, grav_coords, rot_vel), dim=1)
    velocity = torch.cat((pos_vel, grav_coords), dim=1)

    return velocity


def calc_rigid_transform_torch_ver(XX: torch.Tensor, YY: torch.Tensor):
    """ PyTorch implementation of calc_rigid_transform from the physion-particles repository """

    X = XX.t()  # (3, N)
    Y = YY.t()  # (3, N)

    mean_X = X.mean(dim=1, keepdim=True)  # (3,1)
    mean_Y = Y.mean(dim=1, keepdim=True)  # (3,1)

    # not enough points to compute R
    if X.shape[1] < 3:        
        R = torch.eye(3, dtype=X.dtype, device=X.device)
        T = mean_Y - R @ mean_X  # (3,1)
        return R, T

    Xc = X - mean_X
    Yc = Y - mean_Y

    C = Xc @ Yc.t()  # (3,3)
    
    U, S, Vt = torch.linalg.svd(C)  # U:(3,3), S:(3,), Vt:(3,3)

    V = Vt.t()
    det_val = torch.det(V @ U.t())
    s = torch.sign(det_val)  # +1 or -1

    D = torch.eye(3, dtype=XX.dtype, device=XX.device)
    D[-1, -1] = s
    
    R = V @ D @ U.t()  # (3,3)
    T = mean_Y - R @ mean_X  # (3,1)

    return R, T

def get_soft_distribution(scores):

    assert scores.dim() == 3, "Input must be (N, C, M)"
    summed = scores.sum(dim=1)
    probs = F.softmax(summed, dim=1)

    return probs


def get_one_hot_by_majority_vote(scores):

    assert scores.dim() == 3, "scores must be (N, C, M)"
    N, C, M = scores.shape

    # (N, C) mask marking valid cameras (non-zero score vector)
    cam_valid = (scores.abs().sum(dim=2) > 0)  # bool

    # Per-camera winner per point: (N, C)
    cam_winners = scores.argmax(dim=2)

    # Vote counts per object: (N, M)
    one_hot_cam = F.one_hot(cam_winners, num_classes=M)
    one_hot_cam = one_hot_cam * cam_valid.unsqueeze(-1)  # (N, C, M)
    # Count votes per object
    counts = one_hot_cam.sum(dim=1)  # (N, M)

    # Tie-break by total score across cameras: (N, M)
    totals = (scores * cam_valid.unsqueeze(-1)).sum(dim=1)  # (N, M)

    # For each point, keep totals only for objects with max vote count; else -inf
    max_counts, _ = counts.max(dim=1, keepdim=True)            # (N, 1)
    is_tie = counts == max_counts                              # (N, M)
    masked_totals = totals.masked_fill(~is_tie, float("-inf")) # (N, M)

    # Final winners: argmax over masked totals (unique within tie set)
    winners = masked_totals.argmax(dim=1)                      # (N,)

    # One-hot output: (N, M), same dtype/device as input
    onehot = F.one_hot(winners, num_classes=M).to(dtype=scores.dtype, device=scores.device)

    return onehot


def get_one_hot_by_majority_vote_numpy_ver(scores):

    assert scores.ndim == 3
    N, C, M = scores.shape

    # Valid cameras: non-zero score vector
    cam_valid = (np.abs(scores).sum(axis=2) > 0)  # (N, C) bool

    # winner object index per camera per point: (N, C)
    cam_winners = scores.argmax(axis=2)

    # One-hot per camera, then mask invalid cameras
    one_hot_cam = np.eye(M, dtype=np.int64)[cam_winners]  # (N, C, M)
    one_hot_cam = one_hot_cam * cam_valid[..., None]  # (N, C, M)

    # Count votes
    counts = one_hot_cam.sum(axis=1)  # (N, M)

    # Sum scores only from valid cameras
    totals = (scores * cam_valid[..., None]).sum(axis=1)  # (N, M)

    # find the max vote count for each point: (N,1)
    max_counts = counts.max(axis=1, keepdims=True)

    # mask for tie candidates
    is_tie = (counts == max_counts)

    # masked totals: only keep totals for tied objects, others = -inf
    masked_totals = np.where(is_tie, totals, -np.inf)

    # final selected object per point
    winners = masked_totals.argmax(axis=1)  # (N,)

    # one-hot
    onehot = np.eye(M)[winners]  # (N, M)
    
    return onehot.astype(scores.dtype)