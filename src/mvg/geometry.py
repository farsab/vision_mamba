
import numpy as np

def normalize_points(x):
    # x: (N,2)
    mean = x.mean(axis=0)
    std = x.std(axis=0).mean()
    s = np.sqrt(2) / (std + 1e-9)
    T = np.array([[s,0,-s*mean[0]],[0,s,-s*mean[1]],[0,0,1]])
    x_h = np.c_[x, np.ones(len(x))]
    x_n = (T @ x_h.T).T
    return x_n[:,:2], T

def triangulate_linear(P1, P2, x1, x2):
    # P1,P2: 3x4; x1,x2: Nx2
    Xs = []
    for (u1,v1),(u2,v2) in zip(x1, x2):
        A = np.array([u1*P1[2]-P1[0],
                      v1*P1[2]-P1[1],
                      u2*P2[2]-P2[0],
                      v2*P2[2]-P2[1]])
        _,_,Vt = np.linalg.svd(A)
        X = Vt[-1]; X = X/X[3]
        Xs.append(X[:3])
    return np.array(Xs)

def decompose_essential(E):
    U,S,Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:,2]
    return (R1, t), (R1, -t), (R2, t), (R2, -t)
