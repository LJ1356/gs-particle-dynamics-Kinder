"""
Rollout visualizer for the MuJoCo sweep evaluation (qualitative check).

Reads the per-scene files that test_mujoco.py writes:
  params.npz     -> predicted positions, (P, N, 3) in metres
  params_gt.npz  -> GT positions (T, N, 3) + parent_ids (N,)

and renders a side-by-side animation (GT | Predicted), both point clouds
coloured by parent group (cube / drawer / wiper / robot / static), sharing the
same viewpoint and axis limits so drift over the autoregressive rollout is
visible. Output is a GIF (no ffmpeg dependency).

predicted[i] and GT[i] are the same particle in the same order, so the panels
are directly comparable point-for-point.

Usage (from project root):
  PYTHONPATH=. python tools/visualize_mujoco.py \
      --scene_dir rollout_outputs/sweep/mujoco_sweep_epoch50_sweep/demo_6_t0
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v2 as imageio

from tools.dataset_mujoco import PARENT_TO_ID

ID_TO_PARENT = {v: k for k, v in PARENT_TO_ID.items()}


def _load_scene(scene_dir):
    pred = np.load(os.path.join(scene_dir, 'params.npz'))['means3D']        # (P, N, 3)
    gt = np.load(os.path.join(scene_dir, 'params_gt.npz'))
    gt_full = gt['means3D']                                                 # (T, N, 3)
    parent_ids = gt['parent_ids']                                          # (N,)
    # rollout predicts frames 3..T-1; align pred[i] <-> gt_full[3 + i]
    gt_traj = gt_full[3:3 + pred.shape[0]]
    assert gt_traj.shape == pred.shape, f'{gt_traj.shape} vs {pred.shape}'
    return pred, gt_traj, parent_ids


def _colour_setup(parent_ids):
    present = sorted(int(p) for p in np.unique(parent_ids))
    cmap = plt.get_cmap('tab10')
    colour_of = {pid: cmap(i % 10) for i, pid in enumerate(present)}
    point_colours = np.array([colour_of[int(p)] for p in parent_ids])
    legend = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                     markerfacecolor=colour_of[pid], markeredgecolor='none',
                     label=ID_TO_PARENT.get(pid, f'parent_{pid}'))
              for pid in present]
    return point_colours, legend


def _axis_limits(gt_traj, margin=0.05):
    lo = gt_traj.reshape(-1, 3).min(axis=0) - margin
    hi = gt_traj.reshape(-1, 3).max(axis=0) + margin
    return lo, hi


def visualize(scene_dir, out_path, stride=None, fps=10, elev=25, azim=-60, point_size=3):
    pred, gt_traj, parent_ids = _load_scene(scene_dir)
    P = pred.shape[0]
    if stride is None:
        stride = max(1, P // 60)  # aim for ~60 frames
    frame_idx = list(range(0, P, stride))
    if frame_idx[-1] != P - 1:
        frame_idx.append(P - 1)  # always include the final frame

    colours, legend = _colour_setup(parent_ids)
    lo, hi = _axis_limits(gt_traj)

    fig = plt.figure(figsize=(11, 5))
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pr = fig.add_subplot(1, 2, 2, projection='3d')

    def _style(ax, pts, title):
        ax.cla()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colours, s=point_size, depthshade=False)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((hi - lo))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    frames = []
    for t in frame_idx:
        err = float(np.linalg.norm(pred[t] - gt_traj[t], axis=1).mean())
        within5 = float((np.linalg.norm(pred[t] - gt_traj[t], axis=1) <= 0.05).mean())
        _style(ax_gt, gt_traj[t], 'Ground truth')
        _style(ax_pr, pred[t], 'Predicted')
        fig.suptitle(f'{os.path.basename(scene_dir)}   step {t + 1}/{P}   '
                     f'mean err {err*100:.1f} cm   within 5 cm {within5*100:.0f}%',
                     fontsize=11)
        fig.legend(handles=legend, loc='lower center', ncol=len(legend),
                   fontsize=7, frameon=False)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f'Wrote {out_path}  ({len(frames)} frames, {P} rollout steps, '
          f'final mean err {np.linalg.norm(pred[-1] - gt_traj[-1], axis=1).mean()*100:.1f} cm)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='directory containing params.npz and params_gt.npz')
    parser.add_argument('--out', type=str, default=None, help='output gif path')
    parser.add_argument('--stride', type=int, default=None, help='frame subsample step')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--elev', type=float, default=25)
    parser.add_argument('--azim', type=float, default=-60)
    parser.add_argument('--point_size', type=float, default=3)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.scene_dir, 'rollout.gif')
    visualize(args.scene_dir, out_path, stride=args.stride, fps=args.fps,
              elev=args.elev, azim=args.azim, point_size=args.point_size)
