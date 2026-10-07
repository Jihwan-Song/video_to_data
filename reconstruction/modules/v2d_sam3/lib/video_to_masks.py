# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SAM3 tracker as a drop-in for v2d.sam2.lib.video_to_masks.

Same prompts file (Sam2Prompts: one mask, box or point set per object id) and
same output (``<masks_dir>/<obj_id>/<frame:06d>.png``), but tracked by the
SAM2-style tracker inside SAM3 (``Sam3TrackerPredictor``), so a pipeline needs
no SAM2 image. Prompts are propagated backward and forward from their frames.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v2d.common.datatypes import Sam2Prompts
from v2d.common.video import FrameWriter

CHECKPOINT_NAME = "sam3.pt"


def _validate_prompt(prompt) -> None:
    prompt_types = sum([bool(prompt.mask_path), prompt.box is not None, bool(prompt.points)])
    if prompt_types != 1:
        raise ValueError(
            "Each prompt must provide exactly one of mask_path, box, or points. "
            f"Got object_id={prompt.object_id}, frame_index={prompt.frame_index}."
        )


def video_to_masks(
    video_path: str,
    prompts_path: str,
    masks_dir: str,
    weights_dir: str,
    mask_extension: str = "",
) -> None:
    """Track the prompted objects through the video and save one mask per frame and object."""
    from sam3.model.io_utils import load_resource_as_video_frames
    from sam3.model_builder import build_sam3_video_model

    with open(prompts_path) as f:
        prompts = Sam2Prompts.from_dict(json.load(f))
    model = build_sam3_video_model(checkpoint_path=os.path.join(weights_dir, CHECKPOINT_NAME))
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    obj_frames: dict[int, dict[int, np.ndarray]] = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # The tracker's own loader reads mp4 only through decord; load the frames
        # with SAM3's OpenCV loader (same resize and normalisation) instead.
        images, height, width = load_resource_as_video_frames(
            video_path, image_size=predictor.image_size,
            offload_video_to_cpu=True, video_loader_type="cv2",
        )
        state = predictor.init_state(
            video_height=height, video_width=width, num_frames=len(images),
            offload_video_to_cpu=True,
        )
        state["images"] = images
        for prompt in prompts.prompts:
            _validate_prompt(prompt)
            if prompt.mask_path:
                path = Path(prompt.mask_path)
                if not path.is_absolute():
                    path = Path(prompts_path).resolve().parent / path
                mask = torch.from_numpy(np.asarray(Image.open(path).convert("L")) > 0)
                predictor.add_new_mask(
                    inference_state=state, frame_idx=prompt.frame_index,
                    obj_id=prompt.object_id, mask=mask,
                )
                continue
            # The tracker takes coordinates relative to the frame size.
            box = None
            if prompt.box:
                b = prompt.box
                box = np.array([b.x0 / width, b.y0 / height, b.x1 / width, b.y1 / height], np.float32)
            points = labels = None
            if prompt.points:
                points = np.array([[p.x / width, p.y / height] for p in prompt.points], np.float32)
                labels = np.array(prompt.point_labels, np.int32)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=prompt.frame_index,
                obj_id=prompt.object_id, points=points, labels=labels, box=box,
            )

        for reverse in (True, False):
            for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
                state, start_frame_idx=None, max_frame_num_to_track=None,
                reverse=reverse, propagate_preflight=True,
            ):
                for i, obj_id in enumerate(obj_ids):
                    mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    obj_frames.setdefault(int(obj_id), {})[frame_idx] = mask

    for obj_id, frames in obj_frames.items():
        writer = FrameWriter.from_path(Path(masks_dir) / f"{obj_id}{mask_extension}")
        for frame_idx in sorted(frames):
            writer.write_frame(frames[frame_idx], stem=f"{frame_idx:06d}")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track prompted objects through a video with SAM3")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    args = parser.parse_args()
    video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir)
