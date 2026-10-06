"""Score one ego reconstruction against a Track 3 public episode (recon-only check).

The public split ships mocap object poses and scanned meshes, so a reconstruction of a public
video can be scored locally with the submission kit's own metric code (``v2dlb``):

  - CD-O: the kit's Sim(3)-registered Chamfer distance, after the kit's 4096-face budget.
  - Pose metrics (AUC, SP-SR, MP-SR, MPPE) on the reconstructed object trajectory, with the
    same first-frame rigid alignment the Track 3 scorer applies. This is NOT the leaderboard
    number (that scores a policy rollout); it measures how good the reconstruction is as the
    reference the policy will track.

Two pose numbers are reported:
  raw             recon object frame used as-is, exactly as the scorer would see it.
  frame-corrected recon object frame re-expressed in the scanned mesh's frame, using the CD-O
                  mesh registration. The gap between the two shows how much error comes from
                  the reconstructed mesh having a different canonical frame than the scan.

Single object per run for now (multi-object recon is a TODO).

Usage (env ``v2d_kit``):
  python tools/v2d_track3/eval_recon_public.py \
    --recon_dir <run>/result_slam_gravity_aligned --episode 41 --object white_pot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HDD = Path("/PublicHDD4/nvidia2/video_to_data/inputs/v2d_challenge")
FACE_BUDGET = VERTEX_BUDGET = 4096  # config/tracks.json track_3_mesh
AUC_POINTS = 1000


def matrices_to_xyzw(transforms: np.ndarray) -> np.ndarray:
    """(N, 4, 4) -> (N, 7) [x, y, z, qx, qy, qz, qw]."""
    pose = np.empty((len(transforms), 7))
    pose[:, :3] = transforms[:, :3, 3]
    pose[:, 3:] = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    return pose


def gt_mesh_path(dataset_root: Path, name: str) -> Path:
    folder = dataset_root / "mesh" / name
    for candidate in (folder / f"{name}.glb", folder / f"{name}_visual.glb"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no scanned mesh for {name} under {folder}")


def registration(object_mesh, ref, cand, seed):
    """Scale ratio, rotation, translation and candidate centroid of the CD-O Sim(3) registration.

    Mirrors ``object_mesh._object_chamfer_cm``: the candidate is rescaled about its surface
    centroid ``c`` by ``ratio``, then ``x_ref = R x' + t``.
    """
    from v2dlb.mesh_common import _sample_mesh_surface

    _, ref_radius = object_mesh.surface_moments(*ref)
    centroid, cand_radius = object_mesh.surface_moments(*cand)
    ratio = cand_radius / ref_radius
    cand_v = (cand[0] - centroid) / ratio + centroid
    ref_align = _sample_mesh_surface(ref[0], ref[1], object_mesh._ALIGNMENT_SAMPLES, seed + 1)
    cand_align = _sample_mesh_surface(cand_v, cand[1], object_mesh._ALIGNMENT_SAMPLES, seed + 1)
    rotation, translation = object_mesh.align_rigid(ref_align, cand_align)
    return ratio, rotation, translation, centroid


def scan_frame_offset(ratio, rotation, translation, centroid) -> np.ndarray:
    """4x4 transform from the scan's object frame to the recon object frame (scale dropped)."""
    offset = np.eye(4)
    offset[:3, :3] = rotation.T
    offset[:3, 3] = -ratio * rotation.T @ translation + (1.0 - ratio) * centroid
    return offset


def pose_metrics(object_pose, achieved, reference, vertices) -> dict:
    """Kit pose metrics for one object, (T, 7) xyzw against (T, 7) xyzw."""
    achieved = achieved[:, None, None, :]
    reference = reference[:, None, None, :]
    ids = np.zeros(1, dtype=np.int64)
    return {
        metric: object_pose.episode_metric(metric, achieved, reference, ids, vertices[None],
                                           align_initial_pose=True)
        for metric in ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recon_dir", type=Path, required=True,
                        help="result directory holding result.npz and mesh.obj")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--object", required=True, help="object name, e.g. white_pot")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="recon mesh in the object frame (default: <recon_dir>/mesh.obj)")
    parser.add_argument("--dataset_root", type=Path, default=HDD / "track_3/public")
    parser.add_argument("--kit", type=Path, default=HDD / "v2d_submission_kit")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    sys.path.insert(0, str(args.kit))
    from v2dlb import object_mesh, object_pose
    from v2dlb.mesh_budget import budget_mesh
    from v2dlb.mesh_common import _sample_mesh_surface
    from v2dlb.trajectory import episode_parquet, read_episode

    # Reference trajectory, visible frames only.
    gt = read_episode(episode_parquet(args.dataset_root, args.episode), args.episode)
    if args.object not in gt.object_names:
        raise SystemExit(f"{args.object} not in episode {args.episode}: {gt.object_names}")
    slot = gt.object_names.index(args.object)

    # Reconstructed trajectory, valid frames only. Video frame i == parquet row i.
    recon = np.load(args.recon_dir / "result.npz")
    recon_pose = matrices_to_xyzw(recon["object_to_world_transform"])
    recon_valid = recon["object_is_valid"].astype(bool)
    frames = np.intersect1d(gt.frame_index, np.flatnonzero(recon_valid))
    gt_rows = np.searchsorted(gt.frame_index, frames)
    reference = gt.pose_xyzw[gt_rows, slot]

    # CD-O, through the same 4096-face budget the packer applies.
    ref_mesh = budget_mesh(gt_mesh_path(args.dataset_root, args.object), FACE_BUDGET, VERTEX_BUDGET)
    cand_mesh = budget_mesh(args.mesh or args.recon_dir / "mesh.obj", FACE_BUDGET, VERTEX_BUDGET)
    seed = object_mesh.object_seed(args.episode, slot)
    cd_o = object_mesh.object_chamfer_cm(ref_mesh, cand_mesh, seed)
    ratio, rotation, translation, centroid = registration(object_mesh, ref_mesh, cand_mesh, seed)

    # AUC points: the scorer's object-frame samples are not released; sample the scan instead.
    vertices = _sample_mesh_surface(ref_mesh[0], ref_mesh[1], AUC_POINTS, seed)

    offset = scan_frame_offset(ratio, rotation, translation, centroid)
    corrected = matrices_to_xyzw(recon["object_to_world_transform"][frames] @ offset)

    gt_extent = np.ptp(reference[:, :3], axis=0)
    recon_extent = np.ptp(recon_pose[frames, :3], axis=0)
    report = {
        "episode": args.episode,
        "object": args.object,
        "frames": {"video": int(len(recon_valid)), "gt_visible": int(len(gt.frame_index)),
                   "recon_valid": int(recon_valid.sum()), "scored": int(len(frames))},
        "cd_o_cm": cd_o,
        "mesh_size_ratio_recon_over_scan": ratio,
        "trajectory_extent_m": {"gt": gt_extent.tolist(), "recon": recon_extent.tolist()},
        "pose_raw": pose_metrics(object_pose, recon_pose[frames], reference, vertices),
        "pose_frame_corrected": pose_metrics(object_pose, corrected, reference, vertices),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
