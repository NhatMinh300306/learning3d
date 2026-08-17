import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # type: (torch.Tensor, torch.Tensor, float) -> torch.Tensor
        # shapes:
        #   - deltas: (N_rays, N_pts, 1) - Khoảng cách giữa các điểm mẫu kề nhau trên tia
        #   - rays_density: (N_rays, N_pts, 1) - Mật độ (density) tại mỗi điểm mẫu
        #   - returns weights: (N_rays, N_pts, 1) - Trọng số tích hợp của từng điểm mẫu
        # TODO (1.5): Compute transmittance using the equation described in the README
        Ts = [torch.ones((deltas.shape[0], 1), dtype=deltas.dtype, device=deltas.device)]

        for i in range(1, deltas.shape[1]):
            # T[:, i] = T[:, i-1] * torch.exp(-rays_density[:, i-1] * deltas[:, i-1] + eps) # gradient problem due to inplace operation
            Ts.append(Ts[i-1] * torch.exp(-rays_density[:, i-1] * deltas[:, i-1] + eps))

        T = torch.stack(Ts, dim=1)
        weights = T * (1.0 - torch.exp(-rays_density * deltas + eps))

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        # shapes:
        #   - weights: (N_rays, N_pts, 1) - Trọng số tích hợp của từng điểm mẫu
        #   - rays_feature: (N_rays, N_pts, feature_dim) - Đặc trưng (ví dụ: RGB color hoặc high-dim features) dọc theo tia
        #   - returns feature: (N_rays, feature_dim) - Đặc trưng tích hợp (weighted sum)
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        L = weights * rays_feature
        feature = torch.sum(L, dim=1) 

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            # shapes:
            #   - weights: (N_rays, N_pts, 1)
            #   - feature: (N_rays, N_pts, feature_dim)
            #   - returns/renders color/feature: (N_rays, feature_dim)
            feature = self._aggregate(weights=weights, rays_feature=feature.view(-1, n_pts, 3))

            # TODO (1.5): Render depth map
            # shapes:
            #   - weights: (N_rays, N_pts, 1)
            #   - depth_values (sample_lengths): (N_rays, N_pts)
            #   - returns/renders depth: (N_rays, 1)
            depth = self._aggregate(weights=weights, rays_feature=depth_values.view(-1, n_pts, 1))

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
        self.mask_threshold = cfg.mask_threshold
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # shapes:
        #   - implicit_fn: torch.nn.Module (SDF model representing the implicit surface / Hàm ẩn định nghĩa SDF tại điểm query)
        #   - origins: torch.Tensor (N_rays, 3) - Tọa độ điểm bắt đầu của tia sáng (world space)
        #   - directions: torch.Tensor (N_rays, 3) - Hướng của tia sáng (chuẩn hóa)
        #   - returns points: torch.Tensor (N_rays, 3) - Tọa độ các điểm giao với bề mặt
        #   - returns mask: torch.Tensor (N_rays, 1) [bool] - Mặt nạ đánh dấu tia nào giao với bề mặt
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        pts = origins.clone()
        
        for _ in range(self.max_iters):
            sdf = implicit_fn(pts)
            pts = pts + directions * sdf

        mask_sdf = implicit_fn(pts)
        mask_hit = (mask_sdf <= self.mask_threshold) 
        return pts, mask_hit    

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # type: (torch.Tensor, Union[float, torch.Tensor], Union[float, torch.Tensor]) -> torch.Tensor
    # shapes:
    #   - signed_distance: torch.Tensor (N_rays, N_pts, 1) hoặc bất kỳ kích thước (*, 1) - Khoảng cách có dấu (SDF)
    #   - alpha: float hoặc scalar torch.Tensor - Tham số tỷ lệ mật độ (density scale parameter)
    #   - beta: float hoặc scalar torch.Tensor - Tham số kiểm soát độ sắc nét của ranh giới (sharpness parameter)
    #   - returns density: torch.Tensor (cùng shape với signed_distance) - Mật độ tương ứng chuyển từ SDF
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    s = -signed_distance
    psi = torch.where(s <= 0.0,
                      0.5 * torch.exp(s / beta),
                      1 - 0.5 * torch.exp(-s / beta),
            )
    density = alpha * psi
    return density

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            # shapes:
            #   - distance: torch.Tensor (N_rays, N_pts, 1) - SDF tại các điểm mẫu
            #   - color: torch.Tensor (N_rays, N_pts, 3) - Màu sắc tại các điểm mẫu
            #   - density: torch.Tensor (N_rays, N_pts, 1) - Mật độ tương ứng chuyển đổi từ SDF
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q7): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
