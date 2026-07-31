import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    PointLights, RasterizationSettings, MeshRenderer,
    MeshRasterizer, HardPhongShader, TexturesVertex,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings, PointsRenderer, PointsRasterizer,
    AlphaCompositor,
)


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    return parser


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mesh_renderer(image_size=256, device=None):
    if device is None:
        device = get_device()
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device),
    )
    return renderer


def get_points_renderer(image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)):
    if device is None:
        device = get_device()
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius, points_per_pixel=10)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def render_voxel(voxels, image_size=256, device=None, dist=3, elev=10, azim=180):
    """
    voxels: tensor shape (1, 1, Z, Y, X) hoặc (Z, Y, X), giá trị occupancy trong [0, 1]
    """
    if device is None:
        device = get_device()

    voxels = voxels.detach()
    if voxels.dim() == 5:      # (B, 1, Z, Y, X)
        voxels = voxels.squeeze(1)   # -> (B, Z, Y, X)
    elif voxels.dim() == 3:    # (Z, Y, X)
        voxels = voxels.unsqueeze(0)  # -> (1, Z, Y, X)

    voxels_binary = (voxels > 0.5).float()

    mesh = pytorch3d.ops.cubify(voxels_binary, thresh=0.5).to(device)
    verts = mesh.verts_packed()
    color = torch.tensor([0.7, 0.7, 1.0], device=device)
    textures = TexturesVertex(verts[None].clone().fill_(1) * color)
    mesh.textures = textures

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = PointLights(location=[[0, 0, -3]], device=device)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy().clip(0, 1)


def render_pointcloud(points, image_size=256, device=None, dist=3, elev=10, azim=180):
    """
    points: tensor shape (1, N, 3) hoặc (N, 3)
    """
    if device is None:
        device = get_device()

    points = points.detach()
    if points.dim() == 2:
        points = points.unsqueeze(0)

    color = torch.ones_like(points) * torch.tensor([0.3, 0.5, 1.0], device=points.device)
    point_cloud = Pointclouds(points=points, features=color).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    rend = renderer(point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy().clip(0, 1)


def render_mesh(mesh, image_size=256, device=None, dist=3, elev=10, azim=180):
    if device is None:
        device = get_device()

    mesh = mesh.detach() if hasattr(mesh, "detach") else mesh
    verts = mesh.verts_packed()
    color = torch.tensor([0.7, 0.7, 1.0], device=verts.device)
    textures = TexturesVertex(verts[None].clone().fill_(1) * color)
    mesh = mesh.clone()
    mesh.textures = textures
    mesh = mesh.to(device)

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = PointLights(location=[[0, 0, -3]], device=device)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()
    # SỬA: dùng args.device thay vì hardcode 'cpu' để tránh lệch device với mesh_src
    deform_vertices_src = torch.zeros(
        mesh_src.verts_packed().shape, requires_grad=True, device=args.device
    )
    optimizer = torch.optim.Adam([deform_vertices_src], lr=args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)
        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time
        loss_vis = loss.cpu().item()
        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time, iter_time, loss_vis))

    mesh_src.offset_verts_(deform_vertices_src)
    print('Done!')
    return mesh_src


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([pointclouds_src], lr=args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time
        loss_vis = loss.cpu().item()
        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time, iter_time, loss_vis))
    print('Done!')
    return pointclouds_src


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([voxels_src], lr=args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src, voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start.time() if False else time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time, iter_time, loss_vis))

    print('Done!')
    return voxels_src


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
    )

    feed = r2n2_dataset[0]

    # SỬA: không ép toàn bộ tensor về float — faces phải giữ kiểu long (int64)
    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            t = feed[k].to(args.device)
            if k == "faces":
                feed_cuda[k] = t.long()
            else:
                feed_cuda[k] = t.float()

    if args.type == "vox":
        # initialization: voxels_src được huấn luyện như XÁC SUẤT (voxel_loss dùng log() trực tiếp)
        voxels_src = torch.rand(feed_cuda['voxels'].shape, requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        voxels_src = fit_voxel(voxels_src, voxels_tgt, args)

        # ---- visualize ----
        # SỬA: voxels_src đã là xác suất (0-1) do cách huấn luyện, không phải logit
        # nên clamp thay vì áp sigmoid (tránh làm méo giá trị đã học được)
        pred_prob = torch.clamp(voxels_src.detach(), 0.0, 1.0)
        image_pred = render_voxel(pred_prob, image_size=256, device=args.device)
        image_gt = render_voxel(voxels_tgt, image_size=256, device=args.device)

        os.makedirs("output", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_pred); axes[0].set_title("Optimized voxel"); axes[0].axis("off")
        axes[1].imshow(image_gt);   axes[1].set_title("Ground truth voxel"); axes[1].axis("off")
        plt.savefig("output/voxel_comparison.png", bbox_inches="tight")
        print("Saved output/voxel_comparison.png")

    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1, args.n_points, 3], requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        pointclouds_src = fit_pointcloud(pointclouds_src, pointclouds_tgt, args)

        # ---- visualize ----
        image_pred = render_pointcloud(pointclouds_src, image_size=256, device=args.device)
        image_gt = render_pointcloud(pointclouds_tgt, image_size=256, device=args.device)

        os.makedirs("output", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_pred); axes[0].set_title("Optimized point cloud"); axes[0].axis("off")
        axes[1].imshow(image_gt);   axes[1].set_title("Ground truth point cloud"); axes[1].axis("off")
        plt.savefig("output/pointcloud_comparison.png", bbox_inches="tight")
        print("Saved output/pointcloud_comparison.png")

    elif args.type == "mesh":
        # initialization
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        mesh_src = fit_mesh(mesh_src, mesh_tgt, args)

        # ---- visualize ----
        image_pred = render_mesh(mesh_src, image_size=256, device=args.device)
        image_gt = render_mesh(mesh_tgt, image_size=256, device=args.device)

        os.makedirs("output", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_pred); axes[0].set_title("Optimized mesh"); axes[0].axis("off")
        axes[1].imshow(image_gt);   axes[1].set_title("Ground truth mesh"); axes[1].axis("off")
        plt.savefig("output/mesh_comparison.png", bbox_inches="tight")
        print("Saved output/mesh_comparison.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)