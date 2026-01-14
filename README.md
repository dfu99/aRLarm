# aRLarm
Training + playback workflow for a 2D planar robotic arm (tested up to 6-DOF) using SAC with attention.

## Demo
<video src="demo/videos_grid.mp4" controls muted playsinline loop width="100%"></video>

If the video doesn't render, open `demo/videos_grid.mp4`.

## Features
- Planar arm environment with configurable link count and scaling.
- Attention-based policy network (Transformer-style).
- Target modes: mouse, predefined paths, or ball physics target.
- Optional Weights & Biases logging.
- Video capture during play mode.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
Train a model:
```bash
python aRLarm.py --train --steps 50000 --num-links 3
```

Play a model (loads from `./models/` by default):
```bash
python aRLarm.py --play --model rlarm_v7.1[3]_50000_<timestamp>_<runid>.zip
```

Record a video:
```bash
python aRLarm.py --play --record-video --max-frames 250
```

## Common options
- `--target {m,p,b}`: mouse, path, or ball target mode.
- `--target-shape {wavy,star,circle}`: path shape for `--target p`.
- `--num-links N`: number of links (>=2).
- `--window-width`, `--window-height`: render window size.
- `--arm-scale`, `--reference-window`: arm length scaling reference.
- `--wandb` plus `--wandb-project`, `--wandb-entity`, `--wandb-run-name`.

## In-progress / experimental
- Ball physics target mode (`--target b`) is still in flux.
- Jerk regularization flags (`--jerk-penalty`, `--jerk-threshold`) are not stable yet.

## Outputs
- Models are saved under `./models/` (default).
- Optional video output when `--record-video` is enabled.
