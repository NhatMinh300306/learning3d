import torch
import torch.nn.functional as F
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from ray_utils import RayBundle

R, T = look_at_view_transform(dist=3, elev=0, azim=0)

# 2. Tạo camera object
camera = FoVPerspectiveCameras(R=R, T=T)

W, H = 2, 2

x = torch.arange(W)
y = torch.arange(H)

x = (x / W)*2 - 1
y = (y / H)*2 - 1

print(f"x = {x}")
print(f"y = {y}")

xy = torch.meshgrid(y, x, indexing='ij')
print(f"xy = {xy}")

xy_grid = torch.stack(
        tuple( reversed( torch.meshgrid(y, x) ) ),
        dim=-1,
    ).view(W * H, 2)

xy_grid = -xy_grid

ndc_points = xy_grid

ndc_points = torch.cat(
        [
            ndc_points,
            torch.ones_like(ndc_points[..., -1:])
        ],
        dim=-1
    )
print(f"ndc_points = {ndc_points}")

image_plane_points = camera.unproject_points(ndc_points, world_coordinates=True, from_ndc=True)
print(f"image_plane_points = {image_plane_points}")
print(image_plane_points.shape)

# Ở đây, world_points có 4 tọa độ (tương đương cho 4 pixel) -> dùng repeat để  tìm tọa độ của 4 rays chiếu vào 4 pixel
rays_o = camera.get_camera_center().repeat(image_plane_points.shape[0], 1)
print(f"rays_o = {rays_o}")
print(rays_o.shape)

# Lấy tọa độ world - gốc của rays -> hướng của các rays, rồi chuẩn hóa
rays_d = F.normalize(image_plane_points - rays_o)
print(f"rays_d = {rays_d}")
print(rays_d.shape)

# 1.4
rays_o = rays_o.unsqueeze(1).repeat(1, 4, 1)
rays_d = rays_d.unsqueeze(1).repeat(1, 4, 1)

print(rays_o)
print(rays_o.shape)
print(rays_d)
print(rays_d.shape)

z_vals = torch.linspace(1, 10, 4)
z_vals.unsqueeze_(-1).unsqueeze_(0)
print(f"z_vals = {z_vals}")
print(f"z_vals.shape = {z_vals.shape}")

sample_points = rays_o + rays_d * z_vals
print(f"sample_points = {sample_points}")
print(f"sample_points.shape = {sample_points.shape}")

sample_lengths=z_vals * torch.ones_like(sample_points[..., :1])
print(f"sample_lengths = {sample_lengths}")
print(f"sample_lengths.shape = {sample_lengths.shape}")

Ts = torch.ones((sample_points.shape[0], 1))
print(f"Ts = {Ts}")
print(f"Ts.shape = {Ts.shape}")

