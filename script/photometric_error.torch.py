import torch
import numpy as np
class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, h: int, w: int, device):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w
        self.K = torch.tensor(np.array([[fx, 0.0, cx], [0, fy, cy], [0, 0, 1]]), device=device, dtype=torch.float32)
        self._device = device
        self.Kinv = torch.tensor(np.array(
            [[1.0 / fx, 0.0, -cx / fx], [0, 1.0 / fy, -cy / fy], [0, 0, 1]]
        ), device=device, dtype=torch.float32)

    def image_coordinates(self):
        Y, X = torch.meshgrid(torch.arange(self.h), torch.arange(self.w))

        uv = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1)
        uv = uv.reshape(-1, 2)
        return uv.to(self._device)

    def reconstruct(self, uv: torch.tensor, z: torch.tensor):
        uv1 = torch.ones((uv.shape[0], 3), dtype=torch.float32, device=self._device)
        uv1[:, :2] = uv
        return z * torch.matmul(self.Kinv, uv1.t()).t()

    def project(self, pcl):
        uv = torch.matmul(self.K, pcl.t()).t()
        uv = uv/uv[:, 2, None]
        return uv.reshape(-1, 3)[:, :2]

    def select_visible(self, uv: torch.tensor, z: torch.tensor, border=0.01) -> torch.tensor:
        border = max((1, int(border * self.w)))
        return (
            (z > 0)
            & (self.w - border > uv[:, 0])
            & (uv[:, 0] > border)
            & (self.h - border > uv[:, 1])
            & (uv[:, 1] > border)
        )


def compute(cam, I0, Z0, motion, I1, device):

    mask_selected = torch.isfinite(Z0.view(-1, 1)) & (Z0.view(-1, 1) > 0)
    mask_selected = mask_selected.squeeze(1)
    uv0 = cam.image_coordinates()[mask_selected].long()
    pcl0 = cam.reconstruct(
        uv0,
        Z0.view(-1, 1)[mask_selected],
    )

    R = motion[:3, :3]
    t = motion[:3, 3:]
    pcl0t = ((torch.matmul(R, pcl0.t()) + t).t())

    uv0t = cam.project(pcl0t)
    mask_visible = cam.select_visible(uv0t, pcl0t[:, 2])
    uv0t = uv0t[mask_visible].to(torch.int)
    uv0 = uv0[mask_visible].to(torch.int)

    i1x = I1[uv0t[:, 1], uv0t[:, 0]].view(-1)

    i0x = I0[uv0[:, 1], uv0[:, 0]].view(-1)

    r = (i1x.float() - i0x.float())
    return r.mean() / 255


def main():
    import cv2 as cv
    import os
    from datetime import datetime
    import numpy as np
    I0 = cv.imread(f'{os.path.dirname(os.path.abspath(__file__))}/../resource/rgb0.png', cv.IMREAD_GRAYSCALE)
    Z0 = cv.imread(f'{os.path.dirname(os.path.abspath(__file__))}/../resource/depth0.png', cv.IMREAD_ANYDEPTH).astype(float)/5000.0
    I1 = cv.imread(f'{os.path.dirname(os.path.abspath(__file__))}/../resource/rgb1.png', cv.IMREAD_GRAYSCALE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam = Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640, device=device)
    motion = np.identity(4)
    N = 100
    dt = np.zeros((N,))
    for i in range(N):
        t0 = datetime.now()
        error = compute(cam,
            I0=torch.from_numpy(I0).to(device),
            Z0=torch.from_numpy(Z0).to(device).float(),
            motion=torch.from_numpy(motion).to(device).float(),
            I1=torch.from_numpy(I1).to(device),
            device=device)
        t1 = datetime.now()
        dt[i] = (t1-t0).microseconds
        print(f'Execution on took {dt[i]/1e6}s error: [{error}]')
    print(f'Mean = {dt.mean()/1e6} +- {dt.std()/1e6}')


if __name__ == '__main__':
    main()