# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SAM3: find every instance of a text prompt in a video and track it.

Unlike Grounding DINO + SAM2, one prompt returns each matching instance
separately (two identical boards -> two tracks). Writes, under ``output_dir``:

  <object_id>/<frame:06d>.png   binary mask of one instance, frames where present
  instances.json                InstanceTracks: per instance its frames, scores
                                and pixel boxes
"""
import argparse
import os

import numpy as np

from v2d.common.datatypes import BoundingBox, InstanceTrack, InstanceTracks
from v2d.common.video import FrameWriter

CHECKPOINT_NAME = "sam3.pt"


def video_to_instance_masks(
    video_path: str,
    prompt: str,
    output_dir: str,
    weights_dir: str,
    score_threshold: float = 0.5,
) -> InstanceTracks:
    """Detect and track all instances of ``prompt`` from frame 0 to the end."""
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=os.path.join(weights_dir, CHECKPOINT_NAME),
    )
    session_id = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_path,
        # Full-resolution frames of a few hundred frames would otherwise sit on the GPU.
        offload_video_to_cpu=True,
    ))["session_id"]
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt,
        output_prob_thresh=score_threshold,
    ))

    os.makedirs(output_dir, exist_ok=True)
    writers: dict[int, FrameWriter] = {}
    tracks: dict[int, InstanceTrack] = {}
    n_frames = 0
    for response in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
        propagation_direction="forward",
        output_prob_thresh=score_threshold,
    )):
        frame_idx = response["frame_index"]
        out = response["outputs"]
        n_frames = max(n_frames, frame_idx + 1)
        height, width = out["out_binary_masks"].shape[-2:]
        for obj_id, prob, box_xywh, mask in zip(
            out["out_obj_ids"], out["out_probs"],
            out["out_boxes_xywh"], out["out_binary_masks"],
        ):
            obj_id = int(obj_id)
            if obj_id not in tracks:
                writers[obj_id] = FrameWriter.from_path(os.path.join(output_dir, str(obj_id)))
                tracks[obj_id] = InstanceTrack(obj_id, [], [], [])
            writers[obj_id].write_frame(mask.astype(np.uint8) * 255, stem=f"{frame_idx:06d}")
            # SAM3 boxes are xywh relative to the frame size.
            x, y, w, h = (float(v) for v in box_xywh)
            track = tracks[obj_id]
            track.frame_indices.append(frame_idx)
            track.scores.append(float(prob))
            track.boxes.append(BoundingBox(
                x0=x * width, y0=y * height, x1=(x + w) * width, y1=(y + h) * height,
            ))

    for writer in writers.values():
        writer.close()
    predictor.handle_request(dict(type="close_session", session_id=session_id))

    result = InstanceTracks(
        prompt=prompt,
        n_frames=n_frames,
        tracks=[tracks[obj_id] for obj_id in sorted(tracks)],
    )
    result.save(os.path.join(output_dir, "instances.json"))
    print(f"Found {len(result.tracks)} instance(s) of {prompt!r} over {n_frames} frames -> {output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3: track every instance of a text prompt in a video")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    args = parser.parse_args()
    video_to_instance_masks(
        args.video_path, args.prompt, args.output_dir, args.weights_dir,
        score_threshold=args.score_threshold,
    )
