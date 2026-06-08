from pathlib import Path

import os
import torch
import numpy as np
from scipy.signal import bessel
from tools.utils import get_one_hot_by_majority_vote_numpy_ver


def get_largest_dir(path):
    best = None
    for p in path.iterdir():
        if p.is_dir():
            name = p.name
            if name.isdigit():
                val = int(name)
                if best is None or val > best[0]:
                    best = (val, p)

    return best


def get_min_max_dir(path):
    smallest = None
    largest = None

    for p in path.iterdir():
        if not p.is_dir():
            continue

        name = p.name

        if not name.isdigit():
            continue

        val = int(name)

        if smallest is None or val < smallest[0]:
            smallest = (val, p)
        if largest is None or val > largest[0]:
            largest = (val, p)

    return smallest, largest


def visualize(sequence, exp_name, scenario_name, output_root, chamfer_gt_root, delta_gt_root):
    params = dict(np.load(f"{output_root}/{exp_name}/{sequence}/params.npz")) # for my model
    # params = dict(np.load(f"{output_root}/{exp_name}/{sequence}/param_dense.npz"))
    pred_means = torch.tensor(params["means3D"])
    
    A = pred_means[-4]

    fourdgs_dir = f"{chamfer_gt_root}/{scenario_name}/test/{sequence}/gs/"
    smallest_frame, highest_frame = get_min_max_dir(Path(fourdgs_dir))
    smallest_frame = smallest_frame[0]
    highest_frame = highest_frame[0]

    B = np.load(f"{chamfer_gt_root}/{scenario_name}/test/{sequence}/gs/{highest_frame}/params_coarse.npz")
    B = torch.tensor(B["means3D"][0])

    # compute chamfer using entire point clouds (not object-wise)
    assert (pred_means.shape[0] - 3) == (highest_frame - smallest_frame + 1 - 3)
    chamfer = chamfer_dist(A, B).item()
    gt_params = np.load(f"{output_root}/{exp_name}/{sequence}/params_gt.npz")
    A = torch.tensor(gt_params["means3D"][-1])
    B = pred_means[-1]

    # compute delta metrics
    good_count, total_num = avg_delta(B, A, [0.05, 0.1, 0.2])

    B = np.load(f"{chamfer_gt_root}/{scenario_name}/test/{sequence}/gs/{highest_frame}/params_coarse.npz")
    gt_means3D = B["means3D"][0]

    gt_obj_id = np.load(f"{chamfer_gt_root}/{scenario_name}/test/{sequence}/gs/{highest_frame}/gs_soft_ids_coarse.npz")
    gs_soft_ids = gt_obj_id['gaussian_ids_to_object_ids']
    gs_ids = np.expand_dims(np.sum(np.sum(gs_soft_ids, axis=-1), axis=-1), axis=-1)
    idx_sel = np.squeeze(gs_ids != 0.0, axis=-1)
    gs_soft_ids = gs_soft_ids[idx_sel]
    gt_means3D = torch.tensor(gt_means3D[idx_sel])
    gt_obj_ids = get_one_hot_by_majority_vote_numpy_ver(gs_soft_ids)
    gt_obj_ids = torch.tensor(np.argmax(gt_obj_ids, axis=1))

    return chamfer, good_count, total_num


def avg_delta(A, B, thresholds):
    if len(thresholds) == 0:
        raise ValueError("thresholds must contain at least one value")

    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)

    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)

    if A.shape != B.shape:
        raise ValueError(f"A and B must have same shape, got {A.shape} vs {B.shape}")

    dists = torch.norm(A - B, dim=1, p=2)  # shape (N,)
    if dists.numel() == 0:
        return float('nan')

    fractions = []

    for t in thresholds:
        tval = float(t)
        frac_good = (dists <= tval).to(dtype=torch.float32).sum().item()
        fractions.append(frac_good)

    return fractions, A.shape[0]


def chamfer_dist(A, B):
    A = A.float()
    B = B.float()

    dist_matrix = torch.cdist(A, B, p=2)

    min_dist_A_to_B, _ = torch.min(dist_matrix, dim=1)
    min_dist_B_to_A, _ = torch.min(dist_matrix, dim=0)

    chamfer_dist = torch.mean(min_dist_A_to_B) + torch.mean(min_dist_B_to_A)
    return chamfer_dist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name")
    parser.add_argument("--scenario")
    parser.add_argument("--output_root")
    parser.add_argument("--chamfer_gt_root")
    parser.add_argument("--delta_gt_root")
    args = parser.parse_args()

    exp_name = args.exp_name.strip()
    scenario_name = args.scenario.strip()
    output_root = args.output_root.strip()
    chamfer_gt_root = args.chamfer_gt_root.strip()
    delta_gt_root = args.delta_gt_root.strip()
    subdirs = os.listdir(f"{output_root}/" + exp_name)

    all_chamfers = []
    all_gaussians = 0
    all_good_gaussians = [0, 0, 0]

    for subdir in subdirs:
        if "scene_" in subdir:
            sequence = subdir
            print(f"Evaluating {sequence}")
            chamfer, good_gaussians, total_gaussians = visualize(sequence, exp_name, scenario_name, output_root, chamfer_gt_root, delta_gt_root)
            all_chamfers.append(chamfer)
            all_gaussians += total_gaussians
            all_good_gaussians[0] += good_gaussians[0]
            all_good_gaussians[1] += good_gaussians[1]
            all_good_gaussians[2] += good_gaussians[2]

    print(f"Overall mean metrics for {exp_name}")
    print("Chamfer metrics:")
    print(np.mean(all_chamfers))

    all_gaussians += 1e-8
    all_good_gaussians_frac = [g/all_gaussians for g in all_good_gaussians]
    print("Delta metrics:")
    print(np.mean(all_good_gaussians_frac))

    # append the results in a text file where all experiments are recorded
    with open(f"{output_root}/all_metrics.txt", "a") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Chamfer metrics:\n")
        f.write(f"{np.mean(all_chamfers)}\n")
        f.write(f"Delta metrics:\n")
        f.write(f"{np.mean(all_good_gaussians_frac)}\n")

