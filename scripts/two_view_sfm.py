
import argparse, cv2, numpy as np
from mvg.camera import K_from_params
from mvg.features import orb_match
from mvg.estimation import estimate_E_ransac, recover_pose
from mvg.geometry import triangulate_linear
from mvg.io import write_ply_xyzrgb

def make_P(K, R, t):
    Rt = np.hstack([R, t.reshape(3,1)])
    return K @ Rt

def main(args):
    im1 = cv2.imread(args.im1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(args.im2, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise FileNotFoundError('Could not read input images.')
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    pts1, pts2, good, (k1,k2) = orb_match(gray1, gray2)
    if len(pts1) < 8:
        print('Not enough matches.'); return

    K = K_from_params(args.fx, args.fy, args.cx, args.cy)
    E, inl = estimate_E_ransac(pts1, pts2, K)
    pts1i, pts2i = pts1[inl], pts2[inl]

    R, t, inl_pose = recover_pose(E, pts1i, pts2i, K)
    pts1i, pts2i = pts1i[inl_pose], pts2i[inl_pose]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = make_P(K, R, t)

    X = triangulate_linear(P1, P2, pts1i, pts2i)

    # Simple color: pull from im1
    colors = []
    for (u,v) in pts1i:
        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= v_i < im1.shape[0] and 0 <= u_i < im1.shape[1]:
            colors.append(im1[v_i, u_i, ::-1])  # BGR->RGB
        else:
            colors.append([255,255,255])
    colors = np.array(colors, dtype=np.uint8)

    write_ply_xyzrgb(args.out, X, colors)
    print(f'Saved {len(X)} points to {args.out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--im1', required=True)
    ap.add_argument('--im2', required=True)
    ap.add_argument('--fx', type=float, required=True)
    ap.add_argument('--fy', type=float, required=True)
    ap.add_argument('--cx', type=float, required=True)
    ap.add_argument('--cy', type=float, required=True)
    ap.add_argument('--out', default='cloud.ply')
    args = ap.parse_args()
    main(args)
