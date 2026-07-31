import argparse
import os
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--load_checkpoint', action='store_true')
    # SỬA: default sang cpu
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--load_feat', action='store_true')
    return parser


def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh


def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    pred_to_gt_dists2 = knn_pred.dists[..., 0]
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()

    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    gt_to_pred_dists2 = knn_gt.dists[..., 0]
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()

    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics


def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H, W, D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics, pred_points, gt_points


def visualize_step(predictions, mesh_gt, args, step):
    """
    Vẽ và lưu ảnh so sánh dự đoán vs ground truth cho từng loại output.
    Dùng scatter 3D đơn giản (matplotlib) để không phụ thuộc renderer riêng.
    """
    os.makedirs("vis", exist_ok=True)
    fig = plt.figure(figsize=(10, 5))

    if args.type == "vox":
        H, W, D = predictions.shape[2:]
        try:
            vertices_src, faces_src = mcubes.marching_cubes(
                predictions.detach().cpu().squeeze().numpy(), isovalue=0.5
            )
        except Exception:
            plt.close(fig)
            return
        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(vertices_src[:, 0], vertices_src[:, 1], vertices_src[:, 2], s=1, c="steelblue")
        ax.set_title("Predicted voxel (marching cubes)")

        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)[0].cpu().numpy()
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], s=1, c="salmon")
        ax2.set_title("Ground truth mesh")

    elif args.type == "point":
        pred_points = predictions[0].detach().cpu().numpy()
        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)[0].cpu().numpy()

        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], s=1, c="steelblue")
        ax.set_title("Predicted point cloud")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], s=1, c="salmon")
        ax2.set_title("Ground truth point cloud")

    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points)[0].detach().cpu().numpy()
        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)[0].cpu().numpy()

        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], s=1, c="steelblue")
        ax.set_title("Predicted mesh (sampled points)")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], s=1, c="salmon")
        ax2.set_title("Ground truth mesh (sampled points)")

    plt.tight_layout()
    plt.savefig(f"vis/{step}_{args.type}.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_model(args):
    r2n2_dataset = R2N2(
        "test",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    # SỬA: pin_memory chỉ hữu ích khi copy sang GPU
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=use_cuda,
        drop_last=True,
    )
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        # SỬA: map_location đảm bảo load được checkpoint dù được lưu từ GPU
        checkpoint = torch.load(f'checkpoint_{args.type}.pth', map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        with torch.no_grad():
            predictions = model(images_gt, args)

        metrics, pred_points, gt_points_eval = evaluate(predictions, mesh_gt, thresholds, args)

        # Visualization block (đã điền code, trước đây là TODO)
        if (step % args.vis_freq) == 0:
            visualize_step(predictions, mesh_gt, args, step)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (
            step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()
        ))

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score, args)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)