
import numpy as np
import cv2

def solve_pnp(X3d, x2d, K):
    dist = None
    ok, rvec, tvec = cv2.solvePnP(X3d, x2d, K, distCoeffs=dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    return ok, R, t
