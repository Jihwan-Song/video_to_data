# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render every object slot and both hands of a multi-object run side by side.

Reads the result bundles written by ``run_ego_multi_object.py`` (slot 0 in
``<run_dir>/result_slam_gravity_aligned``, slot k in
``<run_dir>/objects/<name>/result_slam_gravity_aligned``). All bundles share
one camera trajectory, so hands are taken from slot 0. Emits two panels,
both rendered from the source camera at the same scale:

    [source frame + projected meshes]  [meshes only, plain background]

Usage:
    python -m v2d.hamer.lib.render_multi_object_video \\
        --run_dir /data/track3_ep000_sam3 \\
        --mano_assets_root /data/weights/wilor/pretrained_models \\
        --output_path /data/track3_ep000_sam3/multi_object_overlay.mp4
"""
import argparse
import json
import os
import subprocess
import tempfile

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw
from tqdm import tqdm

from v2d.hamer.lib.render_hands_aligned_video import (
    _BG_DARKEN,
    _CV_TO_GL_VEC,
    _add_label,
    _add_lights,
    _font,
    _lookat_pose,
    _mano_layer,
    _material,
)

_BUNDLE_DIR = "result_slam_gravity_aligned"
_OBJECT_COLORS = [
    (245, 150, 40),    # orange
    (60, 150, 235),    # blue
    (190, 80, 220),    # purple
    (60, 215, 215),    # cyan
]
_HAND_COLORS = {"left": (70, 200, 110), "right": (245, 100, 180)}  # green, pink


def _load_slots(run_dir: str) -> list[dict]:
    """Slot name, bundle arrays and mesh for every object in ``objects.json``.

    ``result_dir`` in ``objects.json`` is a host path, so bundle locations are
    rebuilt relative to ``run_dir`` instead.
    """
    with open(os.path.join(run_dir, "objects.json")) as f:
        objects = json.load(f)["objects"]
    slots = []
    for obj in sorted(objects, key=lambda o: o["slot"]):
        if obj["slot"] == 0:
            bundle_dir = os.path.join(run_dir, _BUNDLE_DIR)
        else:
            bundle_dir = os.path.join(run_dir, "objects", obj["name"], _BUNDLE_DIR)
        arrays = dict(np.load(os.path.join(bundle_dir, "result.npz")))
        mesh = trimesh.load(os.path.join(bundle_dir, "mesh.obj"), force="mesh", process=False)
        slots.append({
            "name": obj["name"],
            "arrays": arrays,
            "verts": np.asarray(mesh.vertices, dtype=np.float64) * float(arrays["object_scale"]),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
        })
    return slots


def _hand_verts_cam(arrays: dict, side: str, f_idx: int, mano) -> tuple[np.ndarray, np.ndarray]:
    """MANO verts in CV camera space + faces, same construction as ``_build_hand_mesh``."""
    prefix = f"hand_{side}"
    pose_aa = np.concatenate([
        arrays[f"{prefix}_wrist_orient_in_camera"][f_idx],
        arrays[f"{prefix}_finger_pose"][f_idx].reshape(45),
    ]).astype(np.float32)
    betas = arrays[f"{prefix}_betas"].astype(np.float32)
    out = mano(torch.from_numpy(pose_aa)[None], torch.from_numpy(betas)[None])
    verts = out.verts[0].detach().numpy().astype(np.float64)
    faces = mano.th_faces.numpy()
    if side == "left":
        verts[:, 0] *= -1
        faces = faces[:, [0, 2, 1]]
    hand_scale = float(arrays[f"{prefix}_scale"][f_idx])
    if hand_scale != 1.0:
        c = verts.mean(axis=0, keepdims=True)
        verts = (verts - c) * hand_scale + c
    return verts + arrays[f"{prefix}_wrist_trans_in_camera"][f_idx][None, :], faces


def _apply(T: np.ndarray, verts: np.ndarray) -> np.ndarray:
    return verts @ T[:3, :3].T + T[:3, 3][None, :]


def _add_mesh(scene: pyrender.Scene, verts: np.ndarray, faces: np.ndarray, color: tuple) -> None:
    tm = trimesh.Trimesh(verts, faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tm, material=_material(color), smooth=True))


def _add_legend(img: Image.Image, entries: list[tuple[str, tuple]], frame_idx: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _font(16)
    y = 36
    for text, color in entries + [(f"frame {frame_idx}", None)]:
        if color is not None:
            draw.rectangle([8, y + 2, 22, y + 16], fill=color)
        draw.text((28 if color is not None else 8, y), text, fill=(255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        y += 22
    return img


def render_multi_object_video(
    run_dir: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 30.0,
) -> None:
    slots = _load_slots(run_dir)
    hands = slots[0]["arrays"]
    n_frames = int(hands["camera_to_world_transform"].shape[0])
    mano = _mano_layer(mano_assets_root)

    first = Image.open(os.path.join(run_dir, "frames", "000000.png"))
    W_full, H_full = first.size
    pw = (W_full // 2) & ~1
    ph = (H_full // 2) & ~1

    fx, fy, cx, cy = (float(v) for v in hands["camera_intrinsics"])
    sx, sy = pw / W_full, ph / H_full
    cam_src = pyrender.IntrinsicsCamera(fx=fx * sx, fy=fy * sy, cx=cx * sx, cy=cy * sy,
                                        znear=0.01, zfar=20.0)
    renderer = pyrender.OffscreenRenderer(viewport_width=pw, viewport_height=ph)

    legend = [(s["name"], _OBJECT_COLORS[i % len(_OBJECT_COLORS)]) for i, s in enumerate(slots)]
    legend += [(f"{side} hand", color) for side, color in _HAND_COLORS.items()]

    with tempfile.TemporaryDirectory() as tmpdir:
        for f_idx in tqdm(range(n_frames), desc="render", ncols=80):
            # (verts in CV camera space, faces, color) for everything valid this frame.
            meshes_cv = []
            for i, s in enumerate(slots):
                if s["arrays"]["object_is_valid"][f_idx]:
                    o2c = s["arrays"]["object_to_camera_transform"][f_idx].astype(np.float64)
                    meshes_cv.append((_apply(o2c, s["verts"]), s["faces"],
                                      _OBJECT_COLORS[i % len(_OBJECT_COLORS)]))
            for side, color in _HAND_COLORS.items():
                if hands[f"hand_{side}_is_valid"][f_idx]:
                    v, f_ = _hand_verts_cam(hands, side, f_idx, mano)
                    meshes_cv.append((v, f_, color))

            # ---- Left: source frame with projected meshes -----------------
            scene_src = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.18, 0.18, 0.18])
            scene_src.add(cam_src, pose=np.eye(4))
            _add_lights(scene_src, cam_pose=np.eye(4))
            for v, f_, color in meshes_cv:
                _add_mesh(scene_src, v * _CV_TO_GL_VEC, f_, color)
            rgba, _ = renderer.render(scene_src, flags=pyrender.RenderFlags.RGBA)
            bg = Image.open(os.path.join(run_dir, "frames", f"{f_idx:06d}.png")).convert("RGB")
            bg = np.asarray(bg.resize((pw, ph)), dtype=np.float32)
            ov = rgba.astype(np.float32)
            mask = ov[:, :, 3:4] / 255.0
            bg_dim = bg * (1.0 - mask * (1.0 - _BG_DARKEN))
            left = (mask * ov[:, :, :3] + (1.0 - mask) * bg_dim).clip(0, 255).astype(np.uint8)

            # ---- Right: same camera, meshes only ---------------------------
            scene_src.bg_color = np.array([0.78, 0.78, 0.78, 1.0])
            right, _ = renderer.render(scene_src)

            left_img = _add_legend(_add_label(Image.fromarray(left), "video + recon"), legend, f_idx)
            right_img = _add_label(Image.fromarray(right, "RGB"), "recon only")
            grid = Image.new("RGB", (pw * 2, ph))
            grid.paste(left_img, (0, 0))
            grid.paste(right_img, (pw, 0))
            grid.save(os.path.join(tmpdir, f"{f_idx:06d}.png"))

        renderer.delete()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            output_path,
        ], check=True)
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir",          required=True,
                        help="run_ego_multi_object output dir (has objects.json, frames/).")
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    render_multi_object_video(
        run_dir          = args.run_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        fps              = args.fps,
    )


if __name__ == "__main__":
    main()
