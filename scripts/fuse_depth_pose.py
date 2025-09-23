
import argparse, os, json, numpy as np
from glob import glob
from mvg.io import write_ply_xyzrgb

def load_K(json_path):
    with open(json_path,'r') as f:
        d = json.load(f)
    return np.array([[d['fx'],0,d['cx']],[0,d['fy'],d['cy']],[0,0,1]], dtype=float)

def depth_to_points(depth, K):
    h,w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = depth.reshape(-1)
    mask = z > 0
    xs = xs.reshape(-1)[mask]; ys = ys.reshape(-1)[mask]; z = z[mask]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x,y,z], axis=1)
    return pts

def main(args):
    K = load_K(args.K)
    depth_files = sorted(glob(os.path.join(args.depth_dir, 'depth_*.npy')))[::args.stride]
    pose_files = sorted(glob(os.path.join(args.pose_dir, 'pose_*.txt')))[::args.stride]
    assert len(depth_files)==len(pose_files) and len(depth_files)>0, 'Mismatched or empty depth/pose lists.'

    all_pts = []
    for dpath, ppath in zip(depth_files, pose_files):
        depth = np.load(dpath)  # meters
        T = np.loadtxt(ppath)   # 4x4
        pts_c = depth_to_points(depth, K)      # camera frame
        pts_w = (T[:3,:3] @ pts_c.T + T[:3,3:4]).T
        all_pts.append(pts_w)

    X = np.concatenate(all_pts, axis=0)
    write_ply_xyzrgb(args.out, X)
    print(f'Wrote {args.out} with {len(X)} points')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth_dir', required=True)
    ap.add_argument('--pose_dir', required=True)
    ap.add_argument('--K', required=True, help='JSON with fx,fy,cx,cy')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', default='fused_cloud.ply')
    args = ap.parse_args()
    main(args)
