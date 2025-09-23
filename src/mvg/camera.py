
import numpy as np

def K_from_params(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    return K

def pose_rt_to_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t.ravel()
    return T

def invert_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti
