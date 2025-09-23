
import numpy as np

def write_ply_xyzrgb(filename, X, rgb=None):
    n = len(X)
    if rgb is None:
        rgb = np.ones((n,3), dtype=np.uint8)*255
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x,y,z),(r,g,b) in zip(X, rgb):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
