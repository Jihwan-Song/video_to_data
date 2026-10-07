# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the SAM3 checkpoint (gated: request access to facebook/sam3 first)."""
import argparse
import subprocess


def download_weights(output_dir: str):
    subprocess.run([
        "hf", "download",
        "facebook/sam3", "sam3.pt", "config.json",
        "--local-dir", output_dir,
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the SAM3 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_weights(args.output_dir)
