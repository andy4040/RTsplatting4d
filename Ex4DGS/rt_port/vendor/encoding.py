import torch
from torch import nn
import nvdiffrast.torch

class SphMipEncoding(nn.Module):
    def __init__(self, n_levels: int = 8, plane_size: int = 512, feature_dim: int = 16, Sn: int = 1, dim: int = 1, rand_init: bool = False):
        super(SphMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size

        self.register_parameter(
            'fm',
            nn.Parameter(torch.zeros(Sn, dim, plane_size, 2 * plane_size, feature_dim)),
        )

        if rand_init:
            self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level, index=0, weight=False):
        """
        x: [0,1], Nx3
        level: [0, max_level], Nx1
        """
        x[..., 0] = x[..., 0] * 0.5 + 0.25

        decomposed_x = x

        level = torch.broadcast_to(level, decomposed_x.shape[:3]).contiguous()

        fm = self.fm[index]  # [N, L, 2L, feat_dim]

        padding_fm = torch.cat([fm[:, :, self.plane_size :, :], fm, fm[:, :, : self.plane_size, :]], dim=2)
        enc = nvdiffrast.torch.texture(
            padding_fm,
            decomposed_x,
            mip_level_bias=level * self.n_levels,
            boundary_mode='clamp',
            max_mip_level=self.n_levels - 1,
        )

        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                -1,
            )
        )
        return enc
