import numpy as np


class Camera:

    def __init__(self, fx: float, fy: float, cx: float, cy: float, h: int, w: int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w
        self.K = np.array([[fx, 0.0, cx], [0, fy, cy], [0, 0, 1]])

        self.Kinv = np.array(
            [[1.0 / fx, 0.0, -cx / fx], [0, 1.0 / fy, -cy / fy], [0, 0, 1]]
        )

    def resize(self, s: float):
        return Camera(
            self.fx * s, self.fy * s, self.cx * s, self.cy * s, self.h * s, self.w * s
        )

    def image_coordinates(self):
        uv = np.dstack(np.meshgrid(np.arange(self.w), np.arange(self.h)))
        return np.reshape(uv, (-1, 2))

    def reconstruct(self, uv: np.array, z: np.array):
        uv1 = np.ones((uv.shape[0], 3))
        uv1[:, :2] = uv
        return z.reshape((-1, 1)) * (self.Kinv @ uv1.T).T

    def project(self, pcl):
        uv = (self.K @ pcl.T).T
        uv /= uv[:, 2, None]
        return np.reshape(uv, (-1, 3))[:, :2]

    def select_visible(self, uv: np.array, z: np.array, border=0.01) -> np.array:
        border = max((1, int(border * self.w)))
        return (
            (z > 0)
            & (self.w - border > uv[:, 0])
            & (uv[:, 0] > border)
            & (self.h - border > uv[:, 1])
            & (uv[:, 1] > border)
        )


def compute(cam: Camera, I0: np.ndarray, Z0: np.ndarray, motion: np.ndarray, I1: np.ndarray):

    mask_selected = np.isfinite(Z0.reshape(-1, 1)[:, 0]) & (Z0.reshape(-1, 1)[:, 0] > 0)
    uv0 = cam.image_coordinates()[mask_selected]

    pcl0 = cam.reconstruct(
        uv0,
        Z0.reshape(-1, 1)[mask_selected],
    )

    R = motion[:3, :3]
    t = motion[:3, 3:]
    pcl0t = ((R @ pcl0.T) + t).T

    uv0t = cam.project(pcl0t)
    mask_visible = cam.select_visible(uv0t, pcl0t[:, 2])
    uv0t = uv0t[mask_visible].astype(int)

    i1x = I1[uv0t[:, 1], uv0t[:, 0]].reshape((-1,))

    uv0 = cam.image_coordinates()[mask_selected][mask_visible].astype(int)

    i0x = I0[uv0[:, 1], uv0[:, 0]].reshape((-1,))

    r = (i1x - i0x)

    return r.mean()/255


def main():
    import cv2 as cv
    import os
    from datetime import datetime
    I0 = cv.imread(f'{os.path.dirname(os.path.abspath(__file__))}/../resource/rgb0.png', cv.IMREAD_GRAYSCALE)
    Z0 = cv.imread(f'{os.path.dirname(os.path.abspath(__file__))}/../resource/depth0.png', cv.IMREAD_ANYDEPTH).astype(float)/5000.0
    I1 = cv.imread(f'{os.path.dirname(os.path.abspath(__file__))}/../resource/rgb1.png', cv.IMREAD_GRAYSCALE)
    cam = Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640)
    motion = np.identity(4)
    N = 100
    dt = np.zeros((N,))
    for i in range(N):
        t0 = datetime.now()
        error = compute(cam, I0=I0, Z0=Z0, motion=motion, I1=I1)
        t1 = datetime.now()
        dt[i] = (t1-t0).microseconds
        print(f'Execution on took {dt[i]/1e6}s error: [{error}]')
    print(f'Mean = {dt.mean()/1e6} +- {dt.std()/1e6}')

if __name__ == '__main__':
    main()