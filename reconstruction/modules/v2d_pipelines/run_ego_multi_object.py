# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-object egocentric reconstruction for one video.

Slot 0 runs the regular ``run_ego_reconstruction`` pipeline, which also produces
the camera trajectory, hand tracks and calibration. Every further object reruns
only the object branch (SAM3D -> FoundationPose -> EKF) on the same frames,
depth and intrinsics, and is packaged with slot 0's camera and hand tracks.
All result bundles therefore share one world frame and one scale, which the
multi-object pose metrics (RPE, MP-SR) depend on.

Output layout:

  <output_dir>/                  slot 0, identical to run_ego_reconstruction
  <output_dir>/objects/<name>/   object branch and result bundles of slot >= 1
  <output_dir>/objects.json      slot order, names, prompts, final bundle dirs

Every bundle keeps the single-object result.npz schema, so existing consumers
read each one unchanged.

Objects are found by SAM3 instead of Grounding DINO: one text prompt returns
every matching instance separately, so identical objects side by side and a
lid sitting on its pot come apart. SAM3 runs once per distinct prompt on the
(undistorted) video. Each object takes the instance track with the highest
summed score over the video, which drops late false positives such as a
wooden table matching "a wooden board". Objects sharing a prompt are identical
by looks, so their tracks are assigned left to right, on the first frame they
are all seen, in ``--objects`` order; objects.json records that rule.
Different prompts can overlap (the "a white pot" instance also covers its
lid), so a pixel claimed by two objects goes to the smaller, more specific
mask.

Every object, slot 0 included, uses its SAM3 masks directly, overlaps resolved
per frame: SAM2 seeded with them returned the same masks (mean IoU 0.97-0.99
on public episodes 0, 11, 23) and poses well within FoundationPose's own
run-to-run spread. Slot 0's masks reach run_ego_wilor through
``--object_masks_path``, and its hand masks are tracked by the SAM3 tracker
(``--mask_tracker sam3``), so neither Grounding DINO nor SAM2 runs.

Unless ``--reference_frame`` is given, the reference frame is chosen
automatically. Candidates are frames where every object is seen, WiLoR finds a
hand, and no object mask touches the image border (a cut-off object gives a
partial mesh). Each object is scored by its mask solidity (area over convex
hull area) relative to its best solidity over the candidates: a lid lying on
its pot leaves a crescent of the pot, a hand leaves a notch, and neither
depends on how close the object is. A frame scores its least visible object,
and the earliest frame within 0.02 of the best score wins, which favours the
untouched starting layout. The choice is written to
``reference_selection.json``.

Run from ``reconstruction/``. ``--objects`` lists objects in slot order as
``name`` or ``name=prompt`` (the prompt defaults to the name with ``_`` as
spaces). All other flags are passed to run_ego_reconstruction.py::

  python modules/v2d_pipelines/run_ego_multi_object.py \\
    --objects white_pot="a white pot" white_pot_lid="a white pot lid" \\
    --video <video.mp4> --output_dir <out> --hand_tracking hamer \\
    --undistort --run_droid_slam --run_gravity_alignment --export_threejs_result
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from v2d.anycalib.docker.run_video_to_calibration import (
    run_video_to_calibration as run_anycalib_video_to_calibration,
)
from v2d.common.datatypes import BoundingBox, InstanceTrack, InstanceTracks
from v2d.common.result_bundle import (
    gravity_align_result_bundle,
    result_bundle_has_gravity_alignment,
    write_result_bundle,
)
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.pipelines import run_ego_reconstruction as ego
from v2d.pipelines.run_ego_wilor import _apply_sam3d_transform
from v2d.sam3.docker.run_video_to_instance_masks import run_video_to_instance_masks
from v2d.wilor.docker.run_video_to_hands import run_video_to_hands as run_wilor_video
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh

# Track id of the object inside each objects/<name>/masks/ directory.
OBJECT_TRACK_ID = 1


def _parse_object(spec: str) -> tuple[str, str]:
    name, sep, prompt = spec.partition("=")
    if not name:
        raise ValueError(f"Empty object name in --objects entry {spec!r}")
    return name, (prompt if sep else name.replace("_", " "))


def parse_args() -> tuple[argparse.Namespace, list[tuple[str, str]], argparse.Namespace]:
    """Parse --objects, then hand every other flag to run_ego_reconstruction."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="All other flags are run_ego_reconstruction.py flags.",
    )
    p.add_argument(
        "--objects", nargs="+", required=True,
        help="Objects in slot order, each 'name' or 'name=prompt'.",
    )
    p.add_argument(
        "--reference_frame", type=int, default=None,
        help="Use this reference frame instead of choosing one automatically.",
    )
    own, rest = p.parse_known_args()
    objects = [_parse_object(spec) for spec in own.objects]
    names = [name for name, _ in objects]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate object names in --objects: {names}")

    # run_ego_reconstruction.parse_args() reads sys.argv; slot 0 is its object.
    sys.argv = [sys.argv[0], *rest, "--object_prompt", objects[0][1]]
    return ego.parse_args(), objects, own


def _validate_args(args: argparse.Namespace) -> None:
    if args.hand_tracking not in {"hamer", "hawor"}:
        raise ValueError("Multi-object reconstruction supports --hand_tracking hamer or hawor.")
    if args.run_gsplat_refinement:
        # gsplat refines one object and the camera together. A refined camera
        # would differ from the one the other objects are packaged with.
        raise ValueError("--run_gsplat_refinement is not supported with multiple objects.")
    if args.object_mesh is not None:
        raise ValueError("--object_mesh is not supported with multiple objects.")


def _undistort_video(args: argparse.Namespace) -> Path:
    """The video every stage reads; AnyCalib runs here as run_ego_wilor would, then is skipped there."""
    video_path = ego._postprocess_video_path(args)
    if args.undistort:
        anycalib_dir = Path(args.output_dir).resolve() / "anycalib"
        anycalib_dir.mkdir(parents=True, exist_ok=True)
        if not ego._step("AnyCalib undistortion", video_path.exists()):
            run_anycalib_video_to_calibration(
                video_path=args.video,
                intrinsics_path=str(anycalib_dir / "intrinsics.json"),
                distortion_path=str(anycalib_dir / "distortion.json"),
                weights_path=args.anycalib_weights,
                undistorted_video_path=str(video_path),
                undistorted_intrinsics_path=str(anycalib_dir / "undistorted_intrinsics.json"),
                dev=args.dev,
            )
    return video_path


def _resolve_overlaps(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Give each pixel claimed by two objects to the one with the smaller mask.

    The subtraction leaves specks along the smaller mask's rim; parts under 1%
    of the mask are dropped so they do not stretch the box. Larger parts stay
    (a handle split off by the hand holding it).
    """
    resolved = {}
    for name, mask in masks.items():
        smaller = [m for other, m in masks.items() if other != name and m.sum() < mask.sum()]
        if smaller:
            mask = mask & ~np.logical_or.reduce(smaller)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 0.01 * mask.sum()) + 1
            mask = np.isin(labels, keep)
        resolved[name] = mask
    return resolved


def _match_tracks(args: argparse.Namespace, objects: list[tuple[str, str]]) -> dict[str, dict]:
    """Run SAM3 per distinct prompt and give each object one instance track."""
    output_dir = Path(args.output_dir).resolve()
    video_path = _undistort_video(args)
    names_by_prompt: dict[str, list[str]] = {}
    for name, prompt in objects:
        names_by_prompt.setdefault(prompt, []).append(name)

    matched = {}
    for prompt, names in names_by_prompt.items():
        sam3_dir = output_dir / "sam3" / re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_")
        instances_path = sam3_dir / "instances.json"
        if not ego._step(f"SAM3 instances of {prompt!r}", instances_path.exists()):
            run_video_to_instance_masks(
                video_path=str(video_path),
                prompt=prompt,
                output_dir=str(sam3_dir),
                weights_dir=args.sam3_weights,
                dev=args.dev,
            )
        instances = InstanceTracks.load(str(instances_path))
        tracks = sorted(instances.tracks, key=lambda t: sum(t.scores), reverse=True)[:len(names)]
        if len(tracks) < len(names):
            raise RuntimeError(
                f"SAM3 found {len(tracks)} instance(s) of {prompt!r}, "
                f"but {len(names)} object(s) use that prompt: {names}."
            )
        together = sorted(set.intersection(*(set(t.frame_indices) for t in tracks)))
        if not together:
            raise RuntimeError(f"The {len(names)} instances of {prompt!r} are never seen together.")
        tracks.sort(key=lambda t: t.boxes[t.frame_indices.index(together[0])].x0)
        for name, track in zip(names, tracks):
            matched[name] = {
                "track": track,
                "sam3_dir": sam3_dir,
                "n_frames": instances.n_frames,
                "assignment": "left_to_right" if len(names) > 1 else "top_score",
            }
    return matched


def _resolved_masks(
    matched: dict[str, dict], frame: int, shape: tuple[int, int] | None = None
) -> dict[str, np.ndarray]:
    """Every object's SAM3 mask at ``frame``, overlaps resolved.

    An object SAM3 does not see at ``frame`` gets an empty mask of ``shape``
    (or of the other masks' shape).
    """
    masks = {}
    for name, m in matched.items():
        path = m["sam3_dir"] / str(m["track"].object_id) / f"{frame:06d}.png"
        masks[name] = np.array(Image.open(path)) > 0 if path.exists() else None
        if masks[name] is not None:
            shape = masks[name].shape
    masks = {n: mask if mask is not None else np.zeros(shape, bool) for n, mask in masks.items()}
    return _resolve_overlaps(masks)


def _solidity(mask: np.ndarray) -> float:
    hull = cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8)))
    return float(mask.sum()) / max(cv2.contourArea(hull), 1.0)


def _touches_border(mask: np.ndarray, margin: int = 3) -> bool:
    return bool(
        mask[:margin].any() or mask[-margin:].any()
        or mask[:, :margin].any() or mask[:, -margin:].any()
    )


def _select_reference_frame(
    args: argparse.Namespace, matched: dict[str, dict]
) -> tuple[int, dict]:
    """Frame where the least visible object is most visible (see module docstring)."""
    output_dir = Path(args.output_dir).resolve()
    wilor_raw_dir = output_dir / "wilor_raw"
    # Same call as run_ego_wilor, which then skips it.
    if not ego._step("WiLoR per-frame detection + MANO", ego._has_files(wilor_raw_dir)):
        run_wilor_video(
            video_path=str(_undistort_video(args)),
            output_dir=str(wilor_raw_dir),
            weights_dir=args.wilor_weights,
            dev=args.dev,
        )

    def has_hand(frame: int) -> bool:
        path = wilor_raw_dir / f"{frame:06d}.json"
        if not path.exists():
            return False
        with open(path) as f:
            return len(json.load(f)) > 0

    seen = sorted(set.intersection(*(set(m["track"].frame_indices) for m in matched.values())))
    with_hand = [frame for frame in seen if has_hand(frame)]
    if not with_hand:
        raise RuntimeError("No frame shows every object and a hand; pass --reference_frame.")
    solidity, inside = {}, []
    for frame in with_hand:
        masks = _resolved_masks(matched, frame)
        solidity[frame] = {n: _solidity(mask) if mask.any() else 0.0 for n, mask in masks.items()}
        if not any(_touches_border(mask) for mask in masks.values()):
            inside.append(frame)
    # Rather a cut-off object than no reference frame at all.
    candidates = inside or with_hand
    best_solidity = {n: max(solidity[frame][n] for frame in candidates) for n in matched}
    score = {
        frame: min(solidity[frame][n] / best_solidity[n] for n in matched)
        for frame in candidates
    }
    top = max(score.values())
    chosen = next(frame for frame in candidates if score[frame] >= top - 0.02)
    selection = {
        "reference_frame": chosen,
        "score": score[chosen],
        "best_score": top,
        "relative_solidity": {n: solidity[chosen][n] / best_solidity[n] for n in matched},
        "candidate_frames": len(candidates),
        "border_rule_relaxed": not inside,
        "score_at_frame_0": score.get(0),
    }
    return chosen, selection


def _assign_instances(args: argparse.Namespace, matched: dict[str, dict]) -> dict[str, dict]:
    """Reference mask, box and score of every object at the reference frame."""
    ref = args.reference_frame
    for name, m in matched.items():
        if ref not in m["track"].frame_indices:
            raise RuntimeError(f"SAM3 does not see {name} at reference frame {ref}.")
    reference_dir = Path(args.output_dir).resolve() / "sam3" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    assigned = {}
    for name, mask in _resolved_masks(matched, ref).items():
        if not mask.any():
            raise RuntimeError(f"The SAM3 mask of {name} is empty at frame {ref} after overlap removal.")
        track: InstanceTrack = matched[name]["track"]
        ys, xs = np.nonzero(mask)
        mask_path = reference_dir / f"{name}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        assigned[name] = {
            "sam3_object_id": track.object_id,
            "score": track.scores[track.frame_indices.index(ref)],
            "assignment": matched[name]["assignment"],
            "box": BoundingBox(
                x0=float(xs.min()), y0=float(ys.min()), x1=float(xs.max()), y1=float(ys.max()),
            ).to_dict(),
            "mask_path": str(mask_path),
        }
    return assigned


def _write_object_masks(matched: dict[str, dict], name: str, masks_dir: Path) -> None:
    """Write ``name``'s SAM3 masks, overlaps resolved, in SAM2's layout: one PNG per frame."""
    m = matched[name]
    shape = np.array(Image.open(next((m["sam3_dir"] / str(m["track"].object_id)).glob("*.png")))).shape
    masks_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(m["n_frames"]):
        mask = _resolved_masks(matched, frame, shape)[name]
        Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / f"{frame:06d}.png")


def _final_result_suffix(args: argparse.Namespace) -> str:
    """Name of the final bundle directory, as chosen by run_ego_reconstruction."""
    if args.run_gravity_alignment:
        return "result_slam_gravity_aligned" if args.run_droid_slam else "result_gravity_aligned"
    return "result_slam" if args.run_droid_slam else "result"


def _run_object_branch(
    args: argparse.Namespace,
    obj_dir: Path,
    prompt: str,
    instance: dict,
    matched: dict[str, dict],
) -> Path:
    """Run the object branch of run_ego_wilor for one object; return its smoothed poses."""
    output_dir = Path(args.output_dir).resolve()
    video_path = ego._postprocess_video_path(args)
    intrinsics_stable = output_dir / "intrinsics_stable.json"
    ref_rgb = output_dir / "frames" / f"{args.reference_frame:06d}.png"
    ref_depth = output_dir / "depth" / f"{args.reference_frame:06d}.png"

    object_track = obj_dir / "object_track.json"
    masks_dir = obj_dir / "masks"
    object_masks_dir = masks_dir / str(OBJECT_TRACK_ID)
    ref_obj_mask = object_masks_dir / f"{args.reference_frame:06d}.png"
    mesh_dir = obj_dir / "mesh"
    mesh_path = mesh_dir / "textured_mesh.obj"
    mesh_transform = mesh_dir / "mesh_transform.json"
    mesh_intrinsics = mesh_dir / "mesh_intrinsics.json"
    mesh_pretransformed = obj_dir / "mesh_pretransformed.obj"
    mesh_scaled = obj_dir / "mesh_scaled.obj"
    scale_path = obj_dir / "scale.json"
    poses_dir = obj_dir / "poses"
    poses_smooth_dir = obj_dir / "poses_smoothed"
    obj_dir.mkdir(parents=True, exist_ok=True)

    if not object_track.exists():
        with open(object_track, "w") as f:
            json.dump({
                "reference_frame": args.reference_frame,
                "object_id": OBJECT_TRACK_ID,
                "prompt": prompt,
                **instance,
            }, f, indent=2)

    if not ego._step("Write SAM3 masks", ego._has_files(object_masks_dir)):
        _write_object_masks(matched, obj_dir.name, object_masks_dir)

    # Same calls and parameters as the object branch of run_ego_wilor.
    mesh_dir.mkdir(parents=True, exist_ok=True)
    if not ego._step("SAM3D mesh generation", mesh_path.exists()):
        run_image_to_mesh(
            image_path=str(ref_rgb),
            mask_path=str(ref_obj_mask),
            mesh_path=str(mesh_path),
            transform_path=str(mesh_transform),
            intrinsics_path=str(mesh_intrinsics),
            weights_dir=args.sam3d_weights,
            with_texture_baking=True,
            with_mesh_postprocess=True,
            depth_path=str(ref_depth),
            depth_intrinsics_path=str(intrinsics_stable),
            depth_mask_path=str(ref_obj_mask),
            dev=args.dev,
        )

    if not ego._step("Apply SAM3D transform", mesh_pretransformed.exists()):
        _apply_sam3d_transform(str(mesh_path), str(mesh_transform), str(mesh_pretransformed))

    if not ego._step("FoundationPose scale estimation", mesh_scaled.exists()):
        run_estimate_mesh_scale(
            mesh_path=str(mesh_pretransformed),
            rgb_path=str(ref_rgb),
            depth_path=str(ref_depth),
            mask_path=str(ref_obj_mask),
            intrinsics_path=str(intrinsics_stable),
            weights_dir=args.foundation_pose_weights,
            scale_path=str(scale_path),
            rescaled_mesh_path=str(mesh_scaled),
            lo=0.5,
            hi=2.0,
            n_samples=9,
            n_levels=4,
            iou_weight=1.0,
            depth_weight=1.0,
            registration_iterations=5,
            dev=args.dev,
        )

    if not ego._step("FoundationPose tracking", ego._has_files(poses_dir)):
        run_video_to_poses(
            video_path=str(video_path),
            depth_folder=str(output_dir / "depth"),
            masks_folder=str(object_masks_dir),
            camera_intrinsics_path=str(intrinsics_stable),
            mesh_path=str(mesh_scaled),
            poses_dir=str(poses_dir),
            weights_dir=args.foundation_pose_weights,
            reference_frame=args.reference_frame,
            mask_depth=True,
            reregister_iou_thresh=args.reregister_iou_thresh if args.reregister_iou_thresh else None,
            dev=args.dev,
        )

    if not ego._step("EKF pose smoothing", ego._has_files(poses_smooth_dir)):
        run_ekf_smoothing(
            poses_dir=str(poses_dir),
            mesh_path=str(mesh_scaled),
            intrinsics_path=str(intrinsics_stable),
            weights_dir=args.foundation_pose_weights,
            output_dir=str(poses_smooth_dir),
            masks_folder=str(object_masks_dir),
            process_noise_xy=0.01,
            process_noise_z=0.01,
            process_noise_r=0.02,
            measurement_noise_xy=0.01,
            measurement_noise_z=0.04,
            measurement_noise_r=0.02,
            dev=args.dev,
        )

    return poses_smooth_dir


def _package_object(
    args: argparse.Namespace,
    obj_dir: Path,
    name: str,
    prompt: str,
    slot: int,
    poses_dir: Path,
) -> Path:
    """Package one object with slot 0's camera and hands; return its final bundle dir."""
    output_dir = Path(args.output_dir).resolve()
    slot0_result_dir = output_dir / "result"
    with open(slot0_result_dir / "manifest.json") as f:
        shared = json.load(f)["sources"]

    result_dir = obj_dir / "result"
    if not ego._step(f"Package objects/{name}/result/", ego._result_bundle_done(result_dir)):
        write_result_bundle(
            result_dir=str(result_dir),
            frames_dir=shared["frames_dir"],
            intrinsics_path=shared["intrinsics_path"],
            mesh_path=str(obj_dir / "mesh_scaled.obj"),
            object_poses_dir=str(poses_dir),
            object_scale=1.0,
            camera_to_world_dir=shared["camera_to_world_dir"],
            camera_pose_convention=shared["input_camera_pose_convention"],
            left_hand_dir=shared["left_hand_dir"],
            right_hand_dir=shared["right_hand_dir"],
            source_manifest={
                "pipeline": "ego_multi_object",
                "object_name": name,
                "object_prompt": prompt,
                "object_slot": slot,
                "object_track_id": OBJECT_TRACK_ID,
                "shared_camera_and_hands_from": str(slot0_result_dir),
            },
        )

    suffix = _final_result_suffix(args)
    final_dir = obj_dir / suffix
    if suffix == "result":
        return final_dir
    if not ego._result_bundle_done(final_dir):
        shutil.copytree(result_dir, final_dir, dirs_exist_ok=True)
    if args.run_gravity_alignment and not ego._step(
        f"Gravity-align objects/{name}/{suffix}/",
        result_bundle_has_gravity_alignment(str(final_dir), target=args.gravity_align_target),
    ):
        gravity_align_result_bundle(
            result_dir=str(final_dir),
            geocalib_calibration_path=str(ego._run_geocalib_postprocess(args)),
            target=args.gravity_align_target,
        )
    return final_dir


def _check_shared_camera(slot0_dir: Path, other_dir: Path) -> None:
    """All bundles must carry the exact same camera trajectory."""
    with np.load(slot0_dir / "result.npz") as a, np.load(other_dir / "result.npz") as b:
        if not np.array_equal(a["camera_to_world_transform"], b["camera_to_world_transform"]):
            raise RuntimeError(
                f"camera_to_world_transform differs between {slot0_dir} and {other_dir}; "
                "the objects are not in one world frame."
            )


def main() -> None:
    """Run the multi-object reconstruction command-line entrypoint."""
    args, objects, own = parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    matched = _match_tracks(args, objects)
    if own.reference_frame is not None:
        args.reference_frame = own.reference_frame
    else:
        args.reference_frame, selection = _select_reference_frame(args, matched)
        with open(output_dir / "reference_selection.json", "w") as f:
            json.dump(selection, f, indent=2)
        print(f"  Reference frame {args.reference_frame}: score {selection['score']:.3f}")
    instances = _assign_instances(args, matched)
    slot0_name, slot0_prompt = objects[0]
    slot0_masks = output_dir / "sam3" / "masks" / slot0_name
    if not ego._step(f"Write SAM3 masks of {slot0_name}", ego._has_files(slot0_masks)):
        _write_object_masks(matched, slot0_name, slot0_masks)
    args.object_masks_path = str(slot0_masks)
    args.mask_tracker = "sam3"

    print(f"\n=== slot 0: {slot0_name} ({slot0_prompt!r}) ===")
    ego.run_from_args(args)
    slot0_final_dir = output_dir / _final_result_suffix(args)

    index = [{
        "slot": 0, "name": slot0_name, "prompt": slot0_prompt,
        "result_dir": str(slot0_final_dir), **instances[slot0_name],
    }]
    for slot, (name, prompt) in enumerate(objects[1:], start=1):
        print(f"\n=== slot {slot}: {name} ({prompt!r}) ===")
        obj_dir = output_dir / "objects" / name
        poses_dir = _run_object_branch(args, obj_dir, prompt, instances[name], matched)
        final_dir = _package_object(args, obj_dir, name, prompt, slot, poses_dir)
        _check_shared_camera(slot0_final_dir, final_dir)
        index.append({
            "slot": slot, "name": name, "prompt": prompt,
            "result_dir": str(final_dir), **instances[name],
        })

    index_path = output_dir / "objects.json"
    with open(index_path, "w") as f:
        json.dump({"reference_frame": args.reference_frame, "objects": index}, f, indent=2)
    print(f"\nWrote {index_path}")
    for entry in index:
        print(f"  slot {entry['slot']}: {entry['name']:<20} {entry['result_dir']}/")


if __name__ == "__main__":
    main()
