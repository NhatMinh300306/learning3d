import torch
from pytorch3d.loss import mesh_laplacian_smoothing


# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	voxel_src = torch.clamp(voxel_src, min = 1e-7, max = 1.0 - 1e-7)
	loss = -(voxel_tgt * torch.log(voxel_src) + ((1 - voxel_tgt) * torch.log(1 - voxel_src)))
	# implement some loss for binary voxel grids
	return loss.mean()

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# b: batch size

	# Ma trận bình phương khoảng cách mọi cặp điểm trong src và tgt
	distance_matrix = torch.cdist(point_cloud_src, point_cloud_tgt, p=2)**2

	# Chiều S1 -> S2: Với mỗi điểm trong S1, tìm khoảng cách tới điểm gấn nó nhất trong S2
	min_distance_s1_to_s2, _ = torch.min(distance_matrix, dim=2)

	# Chiểu S2 -> S1: Với mỗi điểm trong S2, tìm khoảng cách tới điểm gần nó nhất trong S1
	min_distance_s2_to_s1, _ = torch.min(distance_matrix, dim=1)

	loss_chamfer = torch.mean(min_distance_s1_to_s2) + torch.mean(min_distance_s2_to_s1)
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian

x = torch.Tensor([1, 2, 3])
print(x)