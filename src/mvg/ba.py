
import numpy as np
from scipy.optimize import least_squares
import cv2

def rodrigues_to_rvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()

def rvec_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def project(K, R, t, X):
    Xc = (R @ X.T + t.reshape(3,1)).T
    x = (K @ Xc.T).T
    x = x[:,:2] / x[:,2:3]
    return x

def bundle_adjustment(K, R, t, X, x_obs, fix_intrinsics=True, max_iter=50):
    # Optimize pose (rvec,t) and 3D points X to minimize reprojection error.
    rvec = rodrigues_to_rvec(R)
    params = np.r_[rvec, t, X.ravel()]

    def residuals(p):
        r = p[:3]; tt = p[3:6]; XX = p[6:].reshape(-1,3)
        Rm = rvec_to_R(r)
        xhat = project(K, Rm, tt, XX)
        return (xhat - x_obs).ravel()

    res = least_squares(residuals, params, verbose=0, max_nfev=max_iter)
    r = res.x[:3]; tt = res.x[3:6]; XX = res.x[6:].reshape(-1,3)
    Rm = rvec_to_R(r)
    return Rm, tt, XX, res
