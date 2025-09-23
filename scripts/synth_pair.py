
import argparse, numpy as np
from mvg.camera import K_from_params
from mvg.geometry import decompose_essential, triangulate_linear
from mvg.estimation import estimate_E_ransac, recover_pose
from mvg.io import write_ply_xyzrgb

def simulate(fx=800, fy=800, cx=640, cy=360, n_points=200, baseline=0.2, noise=0.5):
    K = K_from_params(fx, fy, cx, cy)
    # Camera 1 at origin
    R1 = np.eye(3); t1 = np.zeros(3)
    # Camera 2 translated on x, slight yaw
    ang = np.deg2rad(5); R2 = np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    t2 = np.array([baseline, 0, 0])

    # Random 3D points in front of cam1
    X = np.random.uniform([-1,-1,3],[1,1,6], size=(n_points,3))
    def project(R,t,X):
        Xc = (R @ X.T + t.reshape(3,1)).T
        x = (K @ Xc.T).T
        return x[:,:2]/x[:,2:3]

    x1 = project(R1,t1,X)
    x2 = project(R2,t2,X)

    x1 += np.random.randn(*x1.shape)*noise
    x2 += np.random.randn(*x2.shape)*noise

    return K, (R1,t1), (R2,t2), X, x1, x2

def main(args):
    K,(R1,t1),(R2,t2),X_gt,x1,x2 = simulate(n_points=args.n_points, noise=args.noise_px)
    import cv2
    E, inl = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    x1i, x2i = x1[inl.ravel()==1], x2[inl.ravel()==1]
    _, R, t, mask = cv2.recoverPose(E, x1i, x2i, K)
    print('Recovered t (up to scale):', t.ravel())
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, t.reshape(3,1)])
    X_rec = triangulate_linear(P1, P2, x1i, x2i)
    write_ply_xyzrgb('synth_cloud.ply', X_rec)
    print('Saved synth_cloud.ply')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_points', type=int, default=300)
    ap.add_argument('--noise_px', type=float, default=0.5)
    args = ap.parse_args()
    main(args)
