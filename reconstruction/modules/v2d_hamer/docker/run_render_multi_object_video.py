# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hamer.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_multi_object_video(
    run_dir: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 30.0,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hamer.lib.render_multi_object_video",
        inputs={
            "run_dir":          run_dir,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output_path": output_path},
        extra_args={"fps": fps},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render all object slots and hands of a multi-object run (video overlay + world view)")
    parser.add_argument("--run_dir",          required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_multi_object_video(
        run_dir          = args.run_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        fps              = args.fps,
        dev              = args.dev,
    )
