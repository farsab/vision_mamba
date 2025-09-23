
import cv2
import numpy as np

def orb_match(im1, im2, max_kp=3000, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=max_kp)
    k1, d1 = orb.detectAndCompute(im1, None)
    k2, d2 = orb.detectAndCompute(im2, None)
    if d1 is None or d2 is None: 
        return np.empty((0,2)), np.empty((0,2)), [], (k1,k2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    return pts1, pts2, good, (k1,k2)
