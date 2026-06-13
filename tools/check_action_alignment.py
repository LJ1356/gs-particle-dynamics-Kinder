#!/usr/bin/env python
"""Action/state alignment sanity check for the MuJoCo HDF5 dataset.

Determines whether actions_delta_ee_base_10d[t] / actions_delta_ee_transform[t]
corresponds to:
  Alignment A: actions[t]   drives state[t] -> state[t+1]   (drop last action)
  Alignment B: actions[t+1] drives state[t] -> state[t+1]   (drop first action)

Primary evidence: exact SE(3) reconstruction against actions_delta_ee_transform.
Secondary: frame-corrected local xyz comparison against actions_delta_ee_base_10d.
The world-frame xyz comparison is printed only as a labeled diagnostic.

The action translation is expressed in the pre-action end-effector local frame:
    action_xyz[t] = R_t^T @ (p[t+1] - p[t])
so a raw world-frame finite-difference comparison is NOT a valid decision metric.
"""

import argparse
import glob
import os

import h5py
import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Key discovery
# --------------------------------------------------------------------------- #
def _recursive_list(group, prefix=""):
    keys = []
    for k in group.keys():
        obj = group[k]
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(obj, h5py.Group):
            keys.extend(_recursive_list(obj, path))
        else:
            keys.append((path, obj.shape, str(obj.dtype)))
    return keys


def _print_candidates(demo):
    all_keys = _recursive_list(demo)
    for needle in ("action", "ee", "pose", "transform"):
        hits = [k for k in all_keys if needle in k[0].lower()]
        print(f"    candidates containing '{needle}':")
        for path, shape, dtype in hits:
            print(f"      {path}  {shape}  {dtype}")


def _try_load(demo, path):
    """Load a dataset by '/'-joined path, returning None if absent."""
    node = demo
    for part in path.split("/"):
        if part not in node:
            return None
        node = node[part]
    return np.asarray(node)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _inv_se3(T):
    """Batched inverse of (N,4,4) homogeneous transforms."""
    R = T[:, :3, :3]
    p = T[:, :3, 3]
    Rt = np.transpose(R, (0, 2, 1))
    out = np.tile(np.eye(4, dtype=T.dtype), (T.shape[0], 1, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, p)
    return out


def _relative_tf(ee_pose):
    """relative_tf[t] = inv(ee_pose[t]) @ ee_pose[t+1], shape (T-1,4,4)."""
    inv_t = _inv_se3(ee_pose[:-1])
    return np.einsum("nij,njk->nik", inv_t, ee_pose[1:])


def _rotation_angle_deg(R_pred, R_gt):
    """Geodesic angle (deg) between batched rotations, shape (N,)."""
    R_err = np.einsum("nij,njk->nik", np.transpose(R_pred, (0, 2, 1)), R_gt)
    trace = np.trace(R_err, axis1=1, axis2=2)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _se3_metrics(pred, gt):
    """SE(3) error metrics between two (N,4,4) stacks."""
    trans_err = pred[:, :3, 3] - gt[:, :3, 3]
    trans_norm = np.linalg.norm(trans_err, axis=1)
    rot_err = _rotation_angle_deg(pred[:, :3, :3], gt[:, :3, :3])
    frob = np.linalg.norm((pred - gt).reshape(pred.shape[0], -1), axis=1)
    return {
        "n": pred.shape[0],
        "trans_mae": float(np.mean(np.abs(trans_err))),
        "trans_rmse": float(np.sqrt(np.mean(trans_norm ** 2))),
        "trans_max": float(np.max(trans_norm)) if pred.shape[0] else 0.0,
        "rot_mean_deg": float(np.mean(rot_err)) if pred.shape[0] else 0.0,
        "rot_median_deg": float(np.median(rot_err)) if pred.shape[0] else 0.0,
        "rot_max_deg": float(np.max(rot_err)) if pred.shape[0] else 0.0,
        "frob_mean": float(np.mean(frob)),
        "frob_median": float(np.median(frob)),
    }


def _xyz_metrics(pred_xyz, gt_xyz):
    """Frame-corrected local xyz comparison, with zero-motion guards."""
    err = pred_xyz - gt_xyz
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

    pred_n = np.linalg.norm(pred_xyz, axis=1)
    gt_n = np.linalg.norm(gt_xyz, axis=1)
    mask = (pred_n >= EPS) & (gt_n >= EPS)
    n_masked = int((~mask).sum())

    if mask.sum() > 1:
        pm, gm = pred_xyz[mask], gt_xyz[mask]
        cos = np.sum(pm * gm, axis=1) / (np.linalg.norm(pm, axis=1) * np.linalg.norm(gm, axis=1))
        cos_mean = float(np.mean(cos))
        pearson = []
        for d in range(3):
            if np.std(pm[:, d]) < EPS or np.std(gm[:, d]) < EPS:
                pearson.append(float("nan"))
            else:
                pearson.append(float(np.corrcoef(pm[:, d], gm[:, d])[0, 1]))
        pn, gn = pred_n[mask], gt_n[mask]
        norm_corr = (
            float(np.corrcoef(pn, gn)[0, 1])
            if np.std(pn) >= EPS and np.std(gn) >= EPS
            else float("nan")
        )
    else:
        cos_mean, pearson, norm_corr = float("nan"), [float("nan")] * 3, float("nan")

    return {
        "xyz_mae": mae,
        "xyz_rmse": rmse,
        "cos_mean": cos_mean,
        "pearson_xyz": pearson,
        "norm_corr": norm_corr,
        "n_masked": n_masked,
        "n_used": int(mask.sum()),
    }


# --------------------------------------------------------------------------- #
# Per-demo processing
# --------------------------------------------------------------------------- #
def process_demo(demo, demo_name, verbose):
    ee_pose = _try_load(demo, "obs/ee_pose")
    actions_10d = _try_load(demo, "actions_delta_ee_base_10d")
    action_tf = _try_load(demo, "actions_delta_ee_transform")

    if ee_pose is None:
        print(f"  [{demo_name}] ee_pose (obs/ee_pose) NOT FOUND. Candidates:")
        _print_candidates(demo)
        return None
    if actions_10d is None and action_tf is None:
        print(f"  [{demo_name}] no action keys found. Candidates:")
        _print_candidates(demo)
        return None

    ee_pose = ee_pose.astype(np.float64)
    T = ee_pose.shape[0]

    # ---- safety checks ----
    warnings = []
    for name, arr in (("ee_pose", ee_pose), ("actions_10d", actions_10d), ("action_tf", action_tf)):
        if arr is not None and (np.isnan(arr).any() or np.isinf(arr).any()):
            warnings.append(f"{name} contains NaN/Inf")
    if actions_10d is not None and len(actions_10d) != T:
        warnings.append(f"len(actions_10d)={len(actions_10d)} != len(ee_pose)={T}")
    if action_tf is not None and len(action_tf) != T:
        warnings.append(f"len(action_tf)={len(action_tf)} != len(ee_pose)={T}")

    # bottom row + rotation sanity on ee_pose
    bottom = ee_pose[:, 3, :]
    if not np.allclose(bottom, np.array([0, 0, 0, 1.0]), atol=1e-5):
        warnings.append("ee_pose bottom row not [0,0,0,1]")
    R = ee_pose[:, :3, :3]
    dets = np.linalg.det(R)
    if not np.allclose(dets, 1.0, atol=1e-3):
        warnings.append(f"ee_pose rotation det not ~1 (min={dets.min():.4f}, max={dets.max():.4f})")
    ortho = np.einsum("nij,nik->njk", R, R) - np.eye(3)
    if np.abs(ortho).max() > 1e-3:
        warnings.append(f"ee_pose rotations not orthonormal (max dev={np.abs(ortho).max():.2e})")

    if verbose:
        print(f"\n  === demo: {demo_name} ===")
        print(f"  shapes: ee_pose={ee_pose.shape}", end="")
        if actions_10d is not None:
            print(f", actions_10d={actions_10d.shape}", end="")
        if action_tf is not None:
            print(f", action_tf={action_tf.shape}", end="")
        print()
        if actions_10d is not None:
            a = actions_10d.astype(np.float64)
            print("  action 10D per-dim [mean / std / min / max]:")
            for d in range(a.shape[1]):
                print(f"    dim {d}: {a[:,d].mean():+.5f} / {a[:,d].std():.5f} "
                      f"/ {a[:,d].min():+.5f} / {a[:,d].max():+.5f}")
            const = [d for d in range(a.shape[1]) if a[:, d].std() < 1e-6]
            print(f"  near-constant dims (std<1e-6): {const}")
        for w in warnings:
            print(f"  WARNING: {w}")

    out = {"demo": demo_name, "T": T, "warnings": warnings}

    # ---- relative_tf strictly within this demo ----
    rel = _relative_tf(ee_pose)  # (T-1,4,4)
    n_trans = rel.shape[0]
    assert n_trans == T - 1, f"transitions {n_trans} != T-1 {T-1}"

    # local-frame translation gt for xyz check
    p = ee_pose[:, :3, 3]
    world_delta = p[1:] - p[:-1]                       # (T-1,3)
    local_delta = np.einsum("nij,nj->ni", np.transpose(R[:-1], (0, 2, 1)), world_delta)

    # ---- Core check 1: SE(3) reconstruction ----
    if action_tf is not None:
        atf = action_tf.astype(np.float64)
        out["se3_A"] = _se3_metrics(atf[0:T - 1], rel)   # action_tf[t] ~ rel[t]
        out["se3_B"] = _se3_metrics(atf[1:T], rel)       # action_tf[t+1] ~ rel[t]

    # ---- Core check 2: local xyz ----
    if actions_10d is not None:
        axyz = actions_10d[:, :3].astype(np.float64)
        out["xyz_A"] = _xyz_metrics(axyz[0:T - 1], local_delta)
        out["xyz_B"] = _xyz_metrics(axyz[1:T], local_delta)
        # world-frame diagnostic only
        out["world_A"] = _xyz_metrics(axyz[0:T - 1], world_delta)
        out["world_B"] = _xyz_metrics(axyz[1:T], world_delta)

    if verbose:
        if actions_10d is not None:
            print("  first 5 rows actions_10d[:, :3]:")
            print(np.array2string(actions_10d[:5, :3], precision=5, suppress_small=True, prefix="    "))
            print("  first 5 rows local_delta:")
            print(np.array2string(local_delta[:5], precision=5, suppress_small=True, prefix="    "))
            print("  first 5 rows world_delta:")
            print(np.array2string(world_delta[:5], precision=5, suppress_small=True, prefix="    "))
        if action_tf is not None:
            print("  first 3 actions_delta_ee_transform:")
            print(np.array2string(action_tf[:3], precision=4, suppress_small=True, prefix="    "))
            print("  first 3 reconstructed inv(ee_pose[t]) @ ee_pose[t+1]:")
            print(np.array2string(rel[:3], precision=4, suppress_small=True, prefix="    "))

    return out


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #
def _fmt_se3(m):
    return (f"n={m['n']} | trans MAE={m['trans_mae']:.3e} RMSE={m['trans_rmse']:.3e} "
            f"max={m['trans_max']:.3e} | rot mean={m['rot_mean_deg']:.4e} "
            f"med={m['rot_median_deg']:.4e} max={m['rot_max_deg']:.4e} deg | "
            f"frob mean={m['frob_mean']:.3e}")


def _fmt_xyz(m):
    p = m["pearson_xyz"]
    return (f"MAE={m['xyz_mae']:.3e} RMSE={m['xyz_rmse']:.3e} cos={m['cos_mean']:.4f} "
            f"pearson=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] normcorr={m['norm_corr']:.3f} "
            f"(used={m['n_used']}, masked={m['n_masked']})")


def _agg_se3(per_demo, key):
    """Aggregate SE(3): pool RMSE via sqrt of weighted mean-square, others weighted."""
    rows = [d[key] for d in per_demo if key in d]
    if not rows:
        return None
    ns = np.array([r["n"] for r in rows], dtype=np.float64)
    tot = ns.sum()
    rmse = np.sqrt(np.sum(ns * np.array([r["trans_rmse"] ** 2 for r in rows])) / tot)
    return {
        "n": int(tot),
        "trans_mae": float(np.sum(ns * [r["trans_mae"] for r in rows]) / tot),
        "trans_rmse": float(rmse),
        "trans_max": float(max(r["trans_max"] for r in rows)),
        "rot_mean_deg": float(np.sum(ns * [r["rot_mean_deg"] for r in rows]) / tot),
        "rot_median_deg": float(np.median([r["rot_median_deg"] for r in rows])),
        "rot_max_deg": float(max(r["rot_max_deg"] for r in rows)),
        "frob_mean": float(np.sum(ns * [r["frob_mean"] for r in rows]) / tot),
        "frob_median": float(np.median([r["frob_median"] for r in rows])),
    }


def report(per_demo):
    print("\n" + "=" * 78)
    print("PER-DEMO METRICS")
    print("=" * 78)
    have_se3 = any("se3_A" in d for d in per_demo)
    have_xyz = any("xyz_A" in d for d in per_demo)
    for d in per_demo:
        print(f"\n[{d['demo']}] T={d['T']}")
        if "se3_A" in d:
            print(f"  SE(3) A (action_tf[t]):   {_fmt_se3(d['se3_A'])}")
            print(f"  SE(3) B (action_tf[t+1]): {_fmt_se3(d['se3_B'])}")
        if "xyz_A" in d:
            print(f"  local xyz A: {_fmt_xyz(d['xyz_A'])}")
            print(f"  local xyz B: {_fmt_xyz(d['xyz_B'])}")
            print(f"  [world-frame diagnostic only, NOT used for recommendation]")
            print(f"    world A: {_fmt_xyz(d['world_A'])}")
            print(f"    world B: {_fmt_xyz(d['world_B'])}")

    print("\n" + "=" * 78)
    print("AGGREGATED METRICS (across demos)")
    print("=" * 78)
    agg = {}
    if have_se3:
        agg["se3_A"] = _agg_se3(per_demo, "se3_A")
        agg["se3_B"] = _agg_se3(per_demo, "se3_B")
        print(f"  SE(3) A (action_tf[t]):   {_fmt_se3(agg['se3_A'])}")
        print(f"  SE(3) B (action_tf[t+1]): {_fmt_se3(agg['se3_B'])}")
    if have_xyz:
        agg["xyz_A"] = _agg_se3  # placeholder unused
        xa = [d["xyz_A"] for d in per_demo if "xyz_A" in d]
        xb = [d["xyz_B"] for d in per_demo if "xyz_B" in d]
        print(f"  local xyz A: mean RMSE={np.mean([m['xyz_rmse'] for m in xa]):.3e} "
              f"cos={np.nanmean([m['cos_mean'] for m in xa]):.4f}")
        print(f"  local xyz B: mean RMSE={np.mean([m['xyz_rmse'] for m in xb]):.3e} "
              f"cos={np.nanmean([m['cos_mean'] for m in xb]):.4f}")

    # ---- recommendation ----
    print("\n" + "=" * 78)
    print("EXPECTED INTERPRETATION")
    print("=" * 78)
    print("  actions_delta_ee_transform was generated as inv(ee_pose_before) @ ee_pose_after")
    print("  from the SAME ee_pose stream, so the CORRECT alignment should match almost")
    print("  exactly: translation RMSE ~1e-6 and rotation error ~1e-4 deg. The WRONG")
    print("  alignment should be orders of magnitude worse. Decide by the relative gap.")

    print("\n" + "=" * 78)
    print("FINAL RECOMMENDATION")
    print("=" * 78)
    if have_se3:
        a, b = agg["se3_A"], agg["se3_B"]
        # primary: translation RMSE, tiebreak rotation
        a_score = (a["trans_rmse"], a["rot_mean_deg"])
        b_score = (b["trans_rmse"], b["rot_mean_deg"])
        better = "A" if a_score < b_score else "B"
        ratio = (max(a["trans_rmse"], b["trans_rmse"]) /
                 max(min(a["trans_rmse"], b["trans_rmse"]), 1e-12))
        good = min(a["trans_rmse"], b["trans_rmse"]) < 1e-3
        print(f"  SE(3) translation RMSE: A={a['trans_rmse']:.3e}  B={b['trans_rmse']:.3e} "
              f"(relative gap x{ratio:.1f})")
        print(f"  SE(3) rotation mean   : A={a['rot_mean_deg']:.4e}  B={b['rot_mean_deg']:.4e} deg")
        if not good or ratio < 5:
            print("\n  >> INCONCLUSIVE: neither alignment matches well, or the gap is small.")
            print("     Possible causes: wrong ee_pose key, different frame convention,")
            print("     poses sampled before/after control differently, or scaled/clipped actions.")
        elif better == "A":
            print("\n  >> ALIGNMENT A.")
            print("     Use actions[t] for state[t] -> state[t+1]. DROP THE LAST action per episode.")
            print("     Training: transition i pairs (state[i], state[i+1]) with action[i],")
            print("     for i in 0..T-2.")
        else:
            print("\n  >> ALIGNMENT B.")
            print("     Use actions[t+1] for state[t] -> state[t+1]. DROP THE FIRST action per episode.")
            print("     Training: transition i pairs (state[i], state[i+1]) with action[i+1],")
            print("     for i in 0..T-2.")
    else:
        print("  No actions_delta_ee_transform available; rely on local xyz check above.")
    print()


# --------------------------------------------------------------------------- #
def run_file(path, verbose):
    print("\n" + "#" * 78)
    print(f"# FILE: {path}")
    print("#" * 78)
    per_demo = []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            print("  No 'data' group at root. Top-level keys:", list(f.keys()))
            return per_demo
        demo_names = sorted(f["data"].keys())
        print(f"  found {len(demo_names)} demo(s): {demo_names[:5]}"
              f"{' ...' if len(demo_names) > 5 else ''}")
        for name in demo_names:
            res = process_demo(f["data"][name], name, verbose)
            if res is not None:
                res["_file"] = os.path.basename(path)
                per_demo.append(res)
    return per_demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", type=str, default=None)
    ap.add_argument("--root_dir", type=str, default=None)
    ap.add_argument("--quiet", action="store_true", help="suppress per-demo detail dumps")
    args = ap.parse_args()

    if not args.h5_path and not args.root_dir:
        ap.error("provide --h5_path or --root_dir")

    files = []
    if args.h5_path:
        files.append(args.h5_path)
    if args.root_dir:
        files.extend(sorted(glob.glob(os.path.join(args.root_dir, "**", "*.h5"), recursive=True)))
        files.extend(sorted(glob.glob(os.path.join(args.root_dir, "**", "*.hdf5"), recursive=True)))
    files = list(dict.fromkeys(files))

    all_demos = []
    for path in files:
        all_demos.extend(run_file(path, verbose=not args.quiet))

    if not all_demos:
        print("\nNo demos processed; cannot make a recommendation.")
        return
    report(all_demos)


if __name__ == "__main__":
    main()
