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
