import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):  
        # type: (RayBundle) -> RayBundle
        # shapes:
        #   - ray_bundle: RayBundle chứa origins (N_rays, 3) và directions (N_rays, 3)
        #   - returns RayBundle: được cập nhật thêm sample_points (N_rays, N_pts, 3) và sample_lengths (N_rays, N_pts, 1)
        #ray_bundle: RayBundle
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # shapes: z_vals is (n_pts_per_ray,)
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray)

        # TODO (Q1.4): Sample points from z values
        # shapes:
        #   - o: (N_rays, N_pts, 3)
        #   - d: (N_rays, N_pts, 3)
        #   - sample_points: (N_rays, N_pts, 3)
        o = ray_bundle.origins.unsqueeze(1).repeat(1, z_vals.shape[0], 1)
        d = ray_bundle.directions.unsqueeze(1).repeat(1, z_vals.shape[0], 1)
        sample_points = o + d * z_vals.unsqueeze_(-1).unsqueeze_(0)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}