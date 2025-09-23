
import numpy as np
import cv2

def triangulate_cv(P1, P2, x1, x2):
    # x1,x2 are Nx2 pixel coords; returns Nx3
    x1_h = cv2.convertPointsToHomogeneous(x1).reshape(-1,3).T
    x2_h = cv2.convertPointsToHomogeneous(x2).reshape(-1,3).T
    X_h = cv2.triangulatePoints(P1, P2, x1_h[:2], x2_h[:2])
    X = (X_h[:3] / X_h[3]).T
    return X
