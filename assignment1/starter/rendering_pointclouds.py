import numpy as np
import torch
import pytorch3d
import imageio

from starter.render_generic import load_rgbd_data

from starter.utils import (
    get_device,
    get_points_renderer,
    unproject_depth_image,
)

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)


def render_pointcloud_gif():

    device = get_device()

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
        data["cameras1"]
    )

    # Point cloud 2
    points2, rgba2 = unproject_depth_image(
        rgb2,
        mask2,
        depth2,
        data["cameras2"]
    )

    # Union point cloud

    points = torch.cat([points1, points2],dim=0)

    rgba = torch.cat([rgba1, rgba2],dim=0)

    point_cloud = pytorch3d.structures.Pointclouds(
    points=points.unsqueeze(0),
    features=rgba.unsqueeze(0)
)

    renderer = get_points_renderer(image_size=256,radius=0.01,)

    frames = []

    azims = np.linspace(0,360,60,endpoint=False)

    for azim in azims:
        R, T = look_at_view_transform(dist=6,elev=0,azim=azim)

        cameras = FoVPerspectiveCameras(R=R,T=T,device=device)

        rend = renderer(
            point_cloud,
            cameras=cameras
        )

        image = rend[0, ..., :3]

        image = image.cpu().numpy()

        image = np.clip(image,0,1)

        image = (image * 255).astype(np.uint8)

        frames.append(image)

    imageio.mimsave(
        "images/plant_union.gif",
        frames,
        duration=1000 // 15,
        loop=0
    )


if __name__ == "__main__":
    render_pointcloud_gif()