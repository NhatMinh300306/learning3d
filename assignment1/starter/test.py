from starter.render_generic import load_rgbd_data
from starter.utils import unproject_depth_image
import torch

data = load_rgbd_data()

rgb1 = torch.tensor(data["rgb1"]).float()
mask1 = torch.tensor(data["mask1"]).float()
depth1 = torch.tensor(data["depth1"]).float()

rgb2 = torch.tensor(data["rgb2"]).float()
mask2 = torch.tensor(data["mask2"]).float()
depth2 = torch.tensor(data["depth2"]).float()

    # Point cloud 1
points1, rgba1 = unproject_depth_image(
        rgb1,
        mask1,
        depth1,
        data["cameras1"]    )

    # Point cloud 2
points2, rgba2 = unproject_depth_image(
        rgb2,
        mask2,
        depth2,
        data["cameras2"]
    )

points = torch.cat([points1, points2], dim=0)
rgba = torch.cat([rgb1, rgb2], dim=0)

print(points.shape)
print(rgb1.shape)
print(rgb2.shape)
print(rgba.shape)
