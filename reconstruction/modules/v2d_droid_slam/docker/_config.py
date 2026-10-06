# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

IMAGE_NAME = f"{os.environ.get('V2D_IMAGE_PREFIX', '')}v2d_droid_slam:{os.environ.get('V2D_IMAGE_TAG', 'latest')}"
MODULES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
