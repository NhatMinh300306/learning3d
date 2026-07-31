import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import imageio

from pytorch3d.renderer import look_at_view_transform
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_cow_retexturing(cow_path="data/cow.obj", image_size=256, device=None):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    vertices, faces = load_cow_mesh(cow_path)

    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)

    # ==========================
    # Texture gradient
    # ==========================

    color1 = torch.tensor([0.0, 0.0, 1.0])  # Blue
    color2 = torch.tensor([1.0, 0.0, 0.0])  # Red

    z = vertices[0, :, 2]

    z_min = z.min()
    z_max = z.max()

    alpha = (z - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(1)

    textures = (
        alpha * color2
        + (1 - alpha) * color1
    )

    textures = textures.unsqueeze(0)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    mesh = mesh.to(device)

    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0, -3]],
        device=device
    )

    frames = []

    azims = np.linspace(
        0,
        360,
        60,
        endpoint=False
    )

    for azim in azims:

        R, T = look_at_view_transform(
            dist=3,
            elev=20,
            azim=azim
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=60,
            device=device
        )

        image = renderer(
            mesh,
            cameras=cameras,
            lights=lights
        )

        image = image[0, ..., :3].cpu().numpy()
        image = (image * 255).astype(np.uint8)

        frames.append(image)

    imageio.mimsave(
        "images/cow_retexturing.gif",
        frames,
        duration=1000 // 15,
        loop=0,
    )

if __name__ == "__main__":
    render_cow_retexturing()