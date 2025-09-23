## Introduction
- **Two‑view SfM**: ORB features → Essential matrix (RANSAC) → pose recovery → linear triangulation → colored point cloud (PLY).
- **PnP**: Camera re‑localization from 2D–3D matches.
- **RGB‑D pose+depth fusion**: fuse depth frames using known/external poses into a single point cloud.
- **Synthetic pair generator**: sanity‑check geometry with clean correspondences.
- Tiny **bundle‑adjustment (BA)** with `scipy.optimize.least_squares` (optional, small problems).

> Focus: clarity over speed. Few dependencies, runs locally.

## Install
```bash
pip install -r requirements.txt
```

## Quickstart
### 1) Two‑view SfM on your own images
Put two images in `data/sample_pair/` as `im1.jpg, im2.jpg` (roughly same scene).
```bash
python scripts/two_view_sfm.py --im1 data/sample_pair/im1.jpg --im2 data/sample_pair/im2.jpg     --fx 1200 --fy 1200 --cx 960 --cy 540 --out cloud.ply
```

### 2) Synthetic correspondences sanity‑check
```bash
python scripts/synth_pair.py --n_points 300 --noise_px 0.5
```
Prints recovered pose, saves a small PLY (`synth_cloud.ply`).

### 3) Fuse depth + poses into one cloud (RGB‑D or monocular+depth)
Assumes per‑frame `depth_*.npy` (meters), per‑frame `pose_*.txt` (4×4) and intrinsics `K.json`:
```bash
python scripts/fuse_depth_pose.py --depth_dir data/rgbd/depth --pose_dir data/rgbd/poses     --K data/rgbd/K.json --stride 1 --out fused_cloud.ply
```

## Notes
- Intrinsics: pass as `--fx --fy --cx --cy`. If unknown, try EXIF focal length + sensor width or calibrate with a checkerboard.
- PLY is ASCII for easy inspection (CloudCompare, MeshLab).
- For real videos → run feature tracking outside the scope here (e.g., KLT) or extend this repo.
