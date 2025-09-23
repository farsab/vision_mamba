
import cv2
import numpy as np

def estimate_E_ransac(pts1, pts2, K):
    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers = inliers.ravel().astype(bool) if inliers is not None else None
    return E, inliers

def recover_pose(E, pts1, pts2, K):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    mask = mask.ravel().astype(bool)
    return R, t.ravel(), mask

def estimate_F_ransac(pts1, pts2):
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
    inliers = inliers.ravel().astype(bool) if inliers is not None else None
    return F, inliers
