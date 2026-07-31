import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import imageio

from pytorch3d.renderer import look_at_view_transform
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_tetrahedron(image_size = 256, color = [0.7, 0.7, 1], device = None):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    vertices = torch.tensor([[0, 0, 0], 
                            [1, 0, 0], 
                            [0, 2, 0], 
                            [0, 0, 3]], dtype=torch.float32)
    
    faces = torch.tensor([[0, 1, 2],
                         [0, 1, 3], 
                         [0, 2, 3],
                         [1, 2, 3]], dtype=torch.int64)
    
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)


    textures = torch.ones_like(vertices)
    textures = textures * torch.tensor(color)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    mesh = mesh.to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    frames = []

    azims = np.linspace(0, 360, 60, endpoint=False)

    for azim in azims:
        R, T = look_at_view_transform(dist=3, elev=20, azim = azim)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device 
        )

        image = renderer(mesh, cameras=cameras, lights=lights)
        
        image = image[0,..., :3].cpu().numpy()

        image = (image *255).astype(np.uint8)

        frames.append(image)

    imageio.mimsave("images/tetrahedron.gif", frames, duration=1000//15, loop=0)

if __name__ == "__main__":
    render_tetrahedron()

