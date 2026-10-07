# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.sam3.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_instance_masks(
    video_path: str,
    prompt: str,
    output_dir: str,
    weights_dir: str,
    score_threshold: float = 0.5,
    dev: bool = False,
) -> None:
    """Track every instance of ``prompt``; see v2d.sam3.lib.video_to_instance_masks."""
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3.lib.video_to_instance_masks",
        inputs={"video_path": video_path, "weights_dir": weights_dir},
        outputs={"output_dir": output_dir},
        extra_args={"prompt": prompt, "score_threshold": score_threshold},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3: track every instance of a text prompt in a video")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_instance_masks(
        args.video_path, args.prompt, args.output_dir, args.weights_dir,
        score_threshold=args.score_threshold, dev=args.dev,
    )
