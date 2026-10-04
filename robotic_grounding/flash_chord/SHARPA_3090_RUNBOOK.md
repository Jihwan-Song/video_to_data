# Sharpa FlashSAC RTX 3090: Collaborator Runbook

## What this run trains

This run is a physics-based imitation task built around the bundled
`tissue_box_simple` reference sequence. Two floating Sharpa hands learn residual
wrist and finger commands that reproduce the demonstrated tissue-box
pick-and-place motion in simulation. The reference parquet supplies the target
hand and object motion plus contact information; it is not itself a learned
policy.

The reward emphasizes object-pose tracking, then hand-pose and joint tracking,
intended contact support, and smooth, small actions. Episodes terminate when the
wrists or object stray beyond the configured tracking tolerances. During
training, the curriculum gradually removes Virtual Object Control (VOC)
assistance so the final policy must execute the motion through simulated hand
control and contact.

This bundled sequence is a reproducible development and smoke-test task for the
RL pipeline. It should not be confused with the challenge's private evaluation
sequences or treated as a complete training dataset for Track 3.

The following files provide a quick visual check of the task:
- [`robotic_grounding/flash_chord/docs/media/sharpa_*.gif`]
These files can be opened directly from a local clone or viewed on GitHub.

## 0. Assumption
- `flash_chord:latest` has already been built in the shared Docker daemon.

## 1. Connect and check access

```bash
ssh -t <your-user>@147.46.245.249

id
test -w /PublicSSD/$USER
docker ps
docker image inspect flash_chord:latest --format '{{.Id}} {{.Size}}'
nvidia-smi
```

If `docker ps` reports a permission error, ask the server administrator for Docker
access and reconnect after it is granted.

All users share the GPUs. After checking `nvidia-smi` and `docker ps`, select an
idle GPU for the commands below:

```bash
GPU=1
```

## 2. Clone the project

Clone without downloading every Git LFS object, then hydrate only FlashCHORD and
its bundled assets:

```bash
cd /PublicSSD/$USER

GIT_LFS_SKIP_SMUDGE=1 git clone \
  --branch challenge-3-rl \
  https://github.com/Jihwan-Song/video_to_data.git

cd video_to_data
git remote add upstream https://github.com/nvidia-isaac/video_to_data.git

git lfs pull --include="robotic_grounding/flash_chord/**,robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**"
```

Verify the required bundled trajectory:

```bash
git status --short --branch

test -s robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave/77ed746eca2449aeb395e51151f26d3c-0.parquet
```

Docker images are shared by the host and are not copied into your account.

## 3. Floating-hand training algorithms

The repository provides two RL algorithms for the floating Sharpa-hand environment:

| Recipe | Entry point | Description |
| --- | --- | --- |
| `sharpa_flash_sac` | `scripts/train_flash_sac.py` | NVIDIA's validated FlashSAC recipe: 4,096 worlds and approximately 250 million environment transitions. It requires a 48 GiB GPU. |
| `sharpa_ppo` | `scripts/train_rl.py` | The floating-hand PPO baseline, using the same task family with a different learner and training configuration. |

Their experiment definitions are:

```text
src/flash_chord/configs/experiment/sharpa_flash_sac.yaml
src/flash_chord/configs/experiment/sharpa_ppo.yaml
```

Our `sharpa_flash_sac_3090` recipe is not a third algorithm. It inherits
`sharpa_flash_sac` and only reduces the parallel worlds, replay capacity, and
updates per collection so FlashSAC fits on a 24 GiB RTX 3090:

```text
src/flash_chord/configs/experiment/sharpa_flash_sac_3090.yaml
```

The current project run uses `sharpa_flash_sac_3090`.

## 4. Verify the recipe (Optional)

```bash
cd /PublicSSD/$USER/video_to_data/robotic_grounding/flash_chord

docker run --rm --runtime=nvidia --gpus "device=$GPU" \
  -v "$PWD:/workspace" -w /workspace \
  --entrypoint pytest flash_chord:latest \
  tests/test_configuration.py -q
```

Resolve and inspect the complete configuration without training:

```bash
docker run --rm --runtime=nvidia --gpus "device=$GPU" \
  -v "$PWD:/workspace:ro" -w /workspace \
  --entrypoint python flash_chord:latest \
  scripts/train_flash_sac.py \
  experiment=sharpa_flash_sac_3090 \
  task.parquet=/tmp/reference.parquet \
  --cfg job --resolve
```

`sharpa_flash_sac_3090` uses 2,048 worlds, a 2,097,152-entry replay buffer,
four updates per collection, 250,003,456 transitions, and 488,288 configured
optimizer updates.

## 5. Open a development container

```bash
cd /PublicSSD/$USER/video_to_data/robotic_grounding/flash_chord
bash docker/run.sh start latest "$GPU"
```

Inside the container, your checkout is `/workspace` and bundled assets are
`/v2d/assets`. Stop it when finished:

```bash
bash docker/run.sh stop latest "$GPU"
```

Optional: if a long-running command occupies the first container shell, open a
second terminal and enter the same container without interrupting it:

```bash
cd /PublicSSD/$USER/video_to_data/robotic_grounding/flash_chord
bash docker/run.sh shell latest "$GPU"
```

This extra shell is useful for interactive inspection or debugging, but is not
required for setup, training, or evaluation.

The development launcher names containers from the image tag and GPU, so users
cannot run this launcher concurrently on the same GPU. Full runs below use unique,
username-prefixed names.

## 6. Launch an independent full run

Run these commands on the host, not inside another container. Choose a fresh output
directory and container name for every experiment.

```bash
GPU=1
REPO=/PublicSSD/$USER/video_to_data/robotic_grounding/flash_chord
ASSETS=/PublicSSD/$USER/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUTPUT=/PublicSSD/$USER/v2d_outputs/sharpa_full_3090_seed42
CONTAINER=${USER}-sharpa-flash-sac-3090-seed42

test ! -e "$OUTPUT"
test -z "$(docker ps -aq --filter name=^/${CONTAINER}$)"
mkdir -p "$OUTPUT"

docker run -d \
  --name "$CONTAINER" \
  --runtime=nvidia --gpus "device=$GPU" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e MUJOCO_GL=egl \
  -v "$REPO:/workspace" \
  -v "$ASSETS:/v2d/assets:ro" \
  -v "$OUTPUT:/output" \
  -w /workspace \
  --entrypoint python \
  flash_chord:latest \
  -u scripts/train_flash_sac.py \
  experiment=sharpa_flash_sac_3090 \
  "task.parquet='/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave'" \
  reset.seed=42 \
  logging.mode=disabled \
  hydra.run.dir=/output/hydra \
  output_dir=/output
```

Immediately verify the run:

```bash
docker inspect --format \
  'running={{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  "$CONTAINER"

docker logs -f "$CONTAINER"
```

Exit log-following with `Ctrl-C`; training continues. Healthy RTX 3090 steady-state
memory is approximately 20,354 MiB used with 3,763 MiB free.

## 7. Checkpoints, stop, and resume (Optional)

List outputs:

```bash
find "$OUTPUT" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

Checkpoint types:

```text
policy_<steps>.safetensors  actor used for evaluation
state_<steps>.safetensors   learner state used for resume
```

Stop a run with:

```bash
docker stop --time 60 "$CONTAINER"
```

Stopping does not create a new checkpoint. Preserve the logs before removing the
stopped container:

```bash
docker inspect "$CONTAINER"
docker logs "$CONTAINER" > "$OUTPUT/docker.log" 2>&1
docker rm "$CONTAINER"
```

To resume, repeat the launch command with the same recipe, seed, mounts, and output
directory, and append:

```text
training.resume=true
training.checkpoint_path=/output/state_<steps>.safetensors
```

Resume from `state_*.safetensors`, never `policy_*.safetensors`. The replay buffer
is not stored in the checkpoint and is rebuilt after resume.

## 8. Evaluate the final policy

After training exits successfully, confirm the final actor exists and ensure the
selected GPU is otherwise idle:

```bash
test -s "$OUTPUT/policy_250003456.safetensors"
nvidia-smi
```

Then run:

```bash
docker run --rm \
  --runtime=nvidia --gpus "device=$GPU" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e MUJOCO_GL=egl \
  -v "$REPO:/workspace:ro" \
  -v "$ASSETS:/v2d/assets:ro" \
  -v "$OUTPUT:/output" \
  -w /workspace \
  --entrypoint python \
  flash_chord:latest \
  -u scripts/evaluate_policy.py \
  evaluation.checkpoint=/output/policy_250003456.safetensors \
  evaluation.metrics_output=/output/evaluation_4096.json \
  logging.mode=disabled \
  hydra.run.dir=/output/eval_hydra_4096
```

The command uses the original FlashCHORD evaluation default of 4,096 parallel worlds.
The result is `$OUTPUT/evaluation_4096.json`. This evaluation used about 7.2 GiB of
GPU memory in our test.

The official Track 3 submission-kit runner uses the same evaluation code but instead
sets `evaluation.world_count=1` and enables `evaluation.object_trajectories_output` to
produce one trajectory Parquet for each scored episode.

## 9. Share code changes

Work on a personal branch rather than directly on `challenge-3-rl`:

```bash
cd /PublicSSD/$USER/video_to_data
git switch -c <your-user>/<short-topic>

git status --short
git add <files>
git commit -m "Describe the change"
git push -u origin <your-user>/<short-topic>
```

Open a pull request into `challenge-3-rl`. Pushing to the project fork requires
GitHub collaborator access; otherwise push to your own fork and open the pull
request from there.

Inspect later NVIDIA updates without changing the active branch:

```bash
git fetch upstream
git log --oneline challenge-3-rl..upstream/main
```

Do not merge upstream changes into a checkout used by an active experiment until
that run finishes and its commit and resolved configuration have been recorded.

## Shared external data

Future datasets may be shared by the `robotics` group under:

```text
/PublicHDD/jhsong/v2d_assets
```

The bundled tissue-box experiment does not use that directory. Track 3 public data
must first be reconstructed and robot-retargeted before FlashCHORD can train on it.

## Evaluation results

We evaluated the terminal actor
`/PublicSSD/jhsong/v2d_outputs/sharpa_full_3090/policy_250003456.safetensors`
on the bundled `tissue_box_simple` sequence. Its SHA-256 is
`ead7ba166505d67a9ddb3cc2cd56b276b8971d8902766db27577cb8bb6403627`.

| Evaluation | Worlds | CHORD SR | ManipTrans SR | Spider SR | MPPE | Mean ADD | ADD AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary, repository default | 4,096 | 99.44% | 99.93% | 99.95% | 0.596 cm | 0.598 cm | 0.9807 |

The primary report is
`/PublicSSD/jhsong/v2d_outputs/sharpa_full_3090/evaluation_4096.json`.