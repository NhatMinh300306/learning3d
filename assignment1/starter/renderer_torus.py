import numpy as np
import torch
import pytorch3d
import imageio

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)

from starter.utils import (
    get_device,
    get_points_renderer,
)


def render_torus_gif(
    image_size=256,
    num_samples=200,
):

    device = get_device()

    # -------------------------
    # Parametric coordinates
    # -------------------------

    u = torch.linspace(
        0,
        2 * np.pi,
        num_samples
    )

    v = torch.linspace(
        0,
        2 * np.pi,
        num_samples
    )

    U, V = torch.meshgrid(
        u,
        v,
        indexing="ij"
    )

    # -------------------------
    # Torus parameters
    # -------------------------

    R = 1.0
    r = 0.4

    x = (R + r * torch.cos(V)) * torch.cos(U)
    y = (R + r * torch.cos(V)) * torch.sin(U)
    z = r * torch.sin(V)

    points = torch.stack(
        [
            x.flatten(),
            y.flatten(),
            z.flatten(),
        ],
        dim=1
    )

    # -------------------------
    # Color
    # -------------------------

    color = (
        points - points.min()
    ) / (
        points.max() - points.min()
    )

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points],
        features=[color],
    ).to(device)

    renderer = get_points_renderer(
        image_size=image_size,
        radius=0.01,
        device=device,
    )

    frames = []

    azims = np.linspace(
        0,
        360,
        60,
        endpoint=False
    )

    for azim in azims:

        R_cam, T_cam = look_at_view_transform(
            dist=4,
            elev=20,
            azim=azim,
        )

        cameras = FoVPerspectiveCameras(
            R=R_cam,
            T=T_cam,
            device=device,
        )

        rend = renderer(
            point_cloud,
            cameras=cameras,
        )

        image = rend[0, ..., :3]

        image = image.detach().cpu().numpy()

        image = np.clip(
            image,
            0,
            1
        )

        image = (image * 255).astype(np.uint8)

        frames.append(image)

    imageio.mimsave(
        "images/torus.gif",
        frames,
        duration=1000 // 15,
        loop=0,
    )

    print("Saved images/torus.gif")


if __name__ == "__main__":
    render_torus_gif()