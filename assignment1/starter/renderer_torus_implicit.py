import numpy as np
import torch
import mcubes
import imageio
import pytorch3d

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    look_at_view_transform,
)

from starter.utils import (
    get_device,
    get_mesh_renderer,
)


def render_torus_implicit_gif(
    image_size=256,
    voxel_size=64,
):

    device = get_device()

    # ----------------------------------
    # Create voxel grid
    # ----------------------------------

    min_value = -2.0
    max_value = 2.0

    X, Y, Z = torch.meshgrid(
        [
            torch.linspace(
                min_value,
                max_value,
                voxel_size
            )
        ] * 3,
        indexing="ij"
    )

    # ----------------------------------
    # Torus implicit function
    # ----------------------------------

    R = 1.0
    r = 0.4

    voxels = (
        (X**2 + Y**2 + Z**2 + R**2 - r**2) ** 2
        - 4 * R**2 * (X**2 + Y**2)
    )

    # ----------------------------------
    # Marching Cubes
    # ----------------------------------

    vertices, faces = mcubes.marching_cubes(
        mcubes.smooth(voxels.numpy()),
        isovalue=0
    )

    vertices = torch.tensor(
        vertices,
        dtype=torch.float32
    )

    faces = torch.tensor(
        faces.astype(int),
        dtype=torch.int64
    )

    # ----------------------------------
    # Normalize coordinates
    # ----------------------------------

    vertices = (
        vertices / voxel_size
    ) * (
        max_value - min_value
    ) + min_value

    # ----------------------------------
    # Vertex colors
    # ----------------------------------

    textures = (
        vertices - vertices.min()
    ) / (
        vertices.max() - vertices.min()
    )

    textures = pytorch3d.renderer.TexturesVertex(
        textures.unsqueeze(0)
    )

    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=textures,
    ).to(device)

    renderer = get_mesh_renderer(
        image_size=image_size,
        device=device,
    )

    lights = PointLights(
        location=[[0, 0, -4]],
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
            mesh,
            cameras=cameras,
            lights=lights,
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
        "images/torus_implicit.gif",
        frames,
        duration=1000 // 15,
        loop=0,
    )

    print("Saved images/torus_implicit.gif")


if __name__ == "__main__":
    render_torus_implicit_gif()