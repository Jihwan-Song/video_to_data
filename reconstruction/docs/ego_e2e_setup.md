# Ego Reconstruction Setup Guide

Setup for the consolidated entrypoint:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py
```

The older scripts remain available, but new runs should use
`run_ego_reconstruction.py`.

> **Added by jhri626 (2026-10-08, `reconstruction` branch): object detection
> and mask tracking use SAM3 (previously Grounding DINO + SAM2).**
>
> Current object and hand segmentation in `run_ego_reconstruction.py` with
> `--hand_tracking hamer` or `hawor` (defaults `--object_detector sam3
> --mask_tracker sam3`):
>
> 1. SAM3 detects `--object_prompt` in every frame and tracks each instance
>    over the whole video. The instance with the highest summed score is the
>    object; this drops late false positives.
> 2. If that instance is not seen at `--reference_frame`, the run prints a
>    `WARNING` and falls back to Grounding DINO: DINO finds the object box on
>    the reference frame and the mask tracker (SAM3) follows it. SAM3 starts a
>    track only when its detector score reaches 0.7, so an object can be missed
>    from some viewpoints. On public Track 3 episodes 39 and 40 a dust pan
>    lying flat scored below 0.1 and was detected only after it was lifted
>    (frames 97 and 50), while DINO found it at frame 0. With the fallback
>    these two episodes scored the same as Grounding DINO + SAM2 (AUC 0.24 and
>    0.39 vs 0.23 and 0.39).
> 3. The hands are seeded from WiLoR on the reference frame, as before, and
>    tracked by SAM3.
> 4. SAM3D (mesh from the reference-frame mask), FoundationPose, EKF and the
>    rest of the pipeline are unchanged.
>
> Other paths:
>
> - `run_ego_multi_object.py` (step 6) uses the same SAM3 detection for several
>   objects. It has no Grounding DINO fallback yet.
> - `--hand_tracking dynhamr` still uses Grounding DINO + SAM2.
> - The previous behaviour is `--object_detector grounding_dino --mask_tracker
>   sam2`. Keep the Grounding DINO and SAM2 images and weights for it, for the
>   fallback and for DynHaMR.
>
> Evidence (public Track 3, pose AUC against mocap):
>
> - Tracking masks, mesh fixed: SAM3 gave equal or better AUC than SAM2 on
>   episodes 0, 11 and 23 (0.34 → 0.43, 0.25 → 0.31, 0.36 → 0.35), with hand
>   wrist positions within 0.1-0.8 cm (median).
> - SAM3D reference mask, SAM2 vs SAM3, 3 meshes each, mean AUC: ep 0
>   0.42 / 0.25, ep 11 0.32 / 0.34, ep 23 0.24 / 0.07, ep 41 0.28 / 0.51,
>   ep 42 0.56 / 0.54. No consistent difference. SAM3D is not seeded, and
>   about one mesh in three fails (AUC 0.1-0.2) with either mask, which
>   dominates the result.

## Agent Skills

- **Codex:** `ego-reconstruction-setup` prepares the environment;
  `run-ego-reconstruction-video` runs and verifies a video reconstruction.
- **Claude:** `.claude/skills/ego-reconstruction-setup` prepares the environment;
  `.claude/skills/run-ego-reconstruction-video` runs and verifies a video reconstruction.

All commands below run from `reconstruction/`.

## 1. Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-smi` accessible inside containers)
- Python 3.10+
- `ffmpeg` on `PATH`
- A Hugging Face token for SAM3D gated model access when using prompt-based mesh reconstruction

> **Added by jhri626 (2026-10-08, `reconstruction` branch):**
>
> - Request access to `facebook/sam3` on Hugging Face as well. It is gated
>   separately from SAM3D (`facebook/sam-3d-objects`) and is needed by the
>   default HaMeR/HaWoR path and by `run_ego_multi_object.py`.
> - An NVIDIA driver that supports CUDA 12.8 (570 or newer). The `v2d_sam3`
>   image is built on `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`; the other
>   images use `2.5.1-cuda12.4`. Tested with driver 580.95.05.

## 1a. Server-Specific Setup

> **Added by jhri626 (2026-10-08, `reconstruction` branch):** what we did on
> our lab server (shared, RTX 4090 + RTX 3090, 24 GB each, no sudo). Each item
> depends on the server, so check whether it applies before following it.
>
> **Docker and GPU permissions.** Without the `docker` group,
> `/var/run/docker.sock` gives `permission denied`, and nothing can be built
> or run. Ask an administrator to add you, then log in again (SSH, tmux and
> VS Code server included). Use `-aG`: without `-a` your other groups are
> removed.
>
> ```bash
> sudo usermod -aG docker <user>     # required
> sudo usermod -aG vglusers <user>   # optional, see below
> id; docker ps                      # docker must be listed, no permission denied
> ```
>
> Containers get the GPU through the Docker daemon, so `docker` alone is
> enough. Host-side `nvidia-smi` (to watch VRAM during an OOM) needs read
> access to `/dev/nvidia*`; on our server these belong to `vglusers`. Check the
> group with `ls -l /dev/nvidia0`.
>
> **Host Python.** We used a dedicated conda env (Python 3.10) so the host
> packages and the variables below stay out of other environments. Without
> sudo, install git-lfs into the same env. The `--local` hooks then need the
> env to be active for later git commands in this checkout.
>
> ```bash
> conda create -n v2d_recon python=3.10 -y
> conda activate v2d_recon
> conda install -c conda-forge git-lfs -y && git lfs install --local
> ```
>
> **Image names on a shared Docker daemon.** Image names are global to the
> daemon. By default every module builds `v2d_<module>:latest`, which
> overwrites (and later runs) another user's image of the same name. Every ego
> module reads `V2D_IMAGE_PREFIX` and `V2D_IMAGE_TAG`
> (unset: `v2d_<module>:latest`). Pin them in the conda env instead of
> exporting them per shell, so building or running without them is not
> possible. Do not run `docker system prune`, `docker image prune` or
> `docker builder prune` there: they remove other users' images and cache.
> Remove only your own images with `docker rmi <prefix>v2d_<module>:<tag>`.
>
> ```bash
> conda env config vars set -n v2d_recon V2D_IMAGE_PREFIX=nvidia2/ V2D_IMAGE_TAG=v0.4.0
> conda activate v2d_recon     # re-activate to load the variables
> python -c "from v2d.moge.docker._config import IMAGE_NAME; print(IMAGE_NAME)"  # after step 2; expect nvidia2/v2d_moge:v0.4.0
> ```
>
> On a server used only by you, skip this; the default names are fine.
>
> **Disk.** Images go to the Docker data root (`/var/lib/docker` by default),
> which a user cannot move. `docker images` reports 14-26 GB per image, most of
> it shared base layers; check free space there before step 3. Weights take
> about 33 GB and each run's outputs several GB. We kept the checkout on an
> SSD and moved only weights and outputs to an HDD. Link only these two
> subfolders: `data/` itself holds tracked files.
>
> ```bash
> mkdir -p <hdd>/video_to_data/{weights,outputs}
> ln -s <hdd>/video_to_data/weights data/weights
> ln -s <hdd>/video_to_data/outputs data/outputs
> ```
>
> **Per-person accounts.** MANO may not be redistributed, so each person
> registers and downloads it from the MANO website. Hugging Face access
> (`facebook/sam-3d-objects`, `facebook/sam3`) and tokens are also per person;
> the downloaders read `HF_TOKEN` or `~/.cache/huggingface/token`.
>
> **24 GB GPUs.** The repo was validated on 48 GB GPUs. On 24 GB both the
> prompt and the mesh paths, gsplat refinement included, finished without OOM
> on our server. If gsplat runs out of memory, lower
> `--gsplat_refine_batch_size` (default 4) and
> `--gsplat_refine_train_resolution_scale` (default 0.5).
>
> **Choosing a GPU.** Every pipeline stage runs its container with
> `--gpus all`, so it uses the first GPU even when another job already fills
> it (`CUDA_VISIBLE_DEVICES` on the host is not passed into the container).
> Set `V2D_DOCKER_GPUS` to Docker's `--gpus` value to pick one; this lets two
> runs share a two-GPU server. Shell and weight-download wrappers keep
> `--gpus all`.
>
> ```bash
> V2D_DOCKER_GPUS=device=1 python modules/v2d_pipelines/run_ego_reconstruction.py ...
> ```

## 2. Install Host Packages

```bash
./scripts/install_ego_reconstruction_packages.sh
```

## 3. Build Docker Images

```bash
./scripts/build_ego_reconstruction_packages.sh
```

> **Added by jhri626 (2026-10-08, `reconstruction` branch):** the build now
> includes `v2d_sam3`. `--mode hamer` builds every image except HaWoR; this is
> the setup used on our server.

## 4. Download Model Weights

```bash
./scripts/download_ego_reconstruction_weights.sh --accept-nvidia-model-eula
```

The downloader supports narrower modes if you do not want every optional model:

```bash
./scripts/download_ego_reconstruction_weights.sh --mode dynhamr_prompt
./scripts/download_ego_reconstruction_weights.sh --mode hamer_prompt
./scripts/download_ego_reconstruction_weights.sh --mode hamer_mesh
```

SAM3D requires a Hugging Face token for gated model access. Either set `HF_TOKEN`
in your environment or log in with `huggingface-cli login` before downloading or
running SAM3D.

DynHaMR/MANO assets are still manual. Place them here:

```text
data/weights/hand/
├── models/
│   └── MANO_RIGHT.pkl
└── BMC/
    └── *.npy
```

The same MANO layout is used by DynHaMR hand reconstruction and hand alignment.

> **Added by jhri626 (2026-10-08, `reconstruction` branch):**
>
> - The HaMeR and HaWoR modes (and `all`) now also download SAM3 into
>   `data/weights/sam3`, which needs the `facebook/sam3` access from step 1.
>   `dynhamr_prompt` does not. To fetch only SAM3:
>
>   ```bash
>   python -m v2d.sam3.docker.run_download_weights --output_dir data/weights/sam3
>   ```
>
> - `--hand_tracking hamer` also needs MANO, in different places from the
>   DynHaMR layout above. Copy `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` into all
>   three; the WiLoR download only puts `MANO_RIGHT.pkl` directly under
>   `pretrained_models/`, and without `pretrained_models/models/` the run
>   fails with `Can not find MANO assets`:
>
>   ```text
>   data/weights/hamer/_DATA/data/mano/                # HaMeR config
>   data/weights/hamer/_DATA/data/models/              # HaMeR rendering, Three.js export
>   data/weights/wilor/pretrained_models/models/       # WiLoR prompt mask rendering
>   ```
>
> - Our server used `--mode hamer_prompt --accept-nvidia-model-eula`.

## 5. Get The Sample Video

A ready-to-run sample, `assets/airplane.mp4`, ships with the repo via Git LFS.
If it is still a small pointer file, install LFS and pull it:

```bash
git lfs install
git lfs pull --include reconstruction/assets/airplane.mp4
```

Confirm it materialized as a real video file before running the examples:

```bash
ls -lh assets/airplane.mp4
```

## 6. Run The Pipeline

DynHaMR hand tracking with prompt-based SAM3D object reconstruction, DROID-SLAM,
gravity alignment, and Three.js export:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --output_dir data/outputs/airplane_dynhamr \
    --reference_frame 0 \
    --undistort \
    --hand_tracking dynhamr \
    --run_droid_slam \
    --run_gravity_alignment \
    --export_threejs_result \
    --dev
```


> **Agent prompt:** In Claude or Codex, ask: “Run the DynHaMR ego reconstruction pipeline on `assets/airplane.mp4` for a toy airplane with undistortion, DROID-SLAM, gravity alignment, and Three.js export.” The matching run skill supplies this command.

New full pipeline with HaMeR hand tracking, prompt-based SAM3D object
reconstruction, DROID-SLAM, gravity alignment, and gsplat refinement:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --output_dir data/outputs/airplane_hamer \
    --reference_frame 0 \
    --undistort \
    --hand_tracking hamer \
    --run_droid_slam \
    --run_gravity_alignment \
    --run_gsplat_refinement \
    --export_threejs_result \
    --dev
```


> **Agent prompt:** In Claude or Codex, ask: “Run the HaMeR ego reconstruction pipeline on `assets/airplane.mp4` for a toy airplane with undistortion, DROID-SLAM, gravity alignment, gsplat refinement, and Three.js export.” The matching run skill supplies this command.

New full pipeline with a provided object mesh override. The prompt is still used
for object detection, masks, and FoundationPose tracking initialization; the mesh
only replaces SAM3D geometry and scale estimation:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --object_mesh assets/textured_mesh.obj \
    --skip_object_scale_estimation \
    --output_dir data/outputs/airplane_hamer_mesh \
    --reference_frame 0 \
    --undistort \
    --hand_tracking hamer \
    --run_droid_slam \
    --run_gravity_alignment \
    --run_gsplat_refinement \
    --export_threejs_result \
    --dev
```


> **Agent prompt:** In Claude or Codex, ask: “Run HaMeR ego reconstruction on `assets/airplane.mp4` using `assets/textured_mesh.obj` for a toy airplane; skip mesh scale estimation and enable undistortion, DROID-SLAM, gravity alignment, gsplat refinement, and Three.js export.” The matching run skill supplies this command.

> **Added by jhri626 (2026-10-08, `reconstruction` branch):** multi-object
> reconstruction with SAM3. `--objects` lists objects in slot order as `name` or
> `name=prompt`; all other flags go to `run_ego_reconstruction.py`. SAM3 finds
> and tracks the objects and the hands, so Grounding DINO and SAM2 do not run.
> `--run_gsplat_refinement` and `--object_mesh` are not supported here. The
> automatic reference frame can pick a frame where an object is moving, so
> pass `--reference_frame` yourself. See the module docstring for details.
>
> ```bash
> python modules/v2d_pipelines/run_ego_multi_object.py \
>     --objects white_pot="a white pot" white_pot_lid="a white pot lid" \
>     --video <video.mp4> \
>     --output_dir data/outputs/<run_name> \
>     --reference_frame <frame> \
>     --hand_tracking hamer \
>     --undistort \
>     --run_droid_slam \
>     --run_gravity_alignment \
>     --export_threejs_result
> ```

## Legacy Command

The old e2e script is intentionally left untouched. Existing commands like this
still run through the legacy path:

```bash
python modules/v2d_pipelines/run_v2d_ego_e2e.py \
    --video_path assets/airplane.mp4 \
    --prompt "airplane" \
    --output_dir data/outputs/airplane_legacy \
    --depth_source moge
```

## Outputs

The base portable result bundle is written to:

```text
<output_dir>/result/
```

Optional post-processing writes suffixed bundles so stages can be cached and
compared:

```text
<output_dir>/result_slam/
<output_dir>/result_gravity_aligned/
<output_dir>/result_slam_gravity_aligned/
```

When `--export_threejs_result` is enabled, the viewer is written under the final
selected bundle:

```text
<final_result_dir>/threejs_scene/index.html
```

The pipeline is cache-aware. Re-run the same command to resume from completed
stage outputs.
