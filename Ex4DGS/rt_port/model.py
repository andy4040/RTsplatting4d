"""RT-Splatting materials attached to Ex4DGS's existing Gaussian collection.

Adapted from RT-Splatting scene/gaussian_model.py at upstream commit
3f45b3cac4be04db9f3092234666b695991b268a; see vendor provenance and licenses.

The SphMip representation, light MLP, material activations and optimizer rates
follow the pinned official RT-Splatting implementation. This adapter does not
own geometry, add planes, or freeze Ex4DGS parameters. Its material rows follow
the static-then-dynamic ordering returned by the Ex4DGS model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def quaternion_product(left, right):
    """Hamilton product for the (w,x,y,z) convention shared by both renderers."""
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((lw * rw - lx * rx - ly * ry - lz * rz,
                        lw * rx + lx * rw + ly * rz - lz * ry,
                        lw * ry - lx * rz + ly * rw + lz * rx,
                        lw * rz + lx * ry - ly * rx + lz * rw), dim=-1)


def initial_surfel_axes(scales):
    """Choose a proper cyclic frame whose normal is the shortest initial axis.

    The choice is stored once, rather than taking argmin every training step:
    a discontinuous axis switch would change the material's surface identity.
    Subsequent tangent scales and frame rotations remain differentiable.
    """
    normal_axis = scales.detach().argmin(dim=-1)
    tangent_axes = torch.stack(((normal_axis + 1) % 3,
                                (normal_axis + 2) % 3), dim=-1)
    # Columns [y,z,x], [z,x,y], and [x,y,z], all with determinant +1.
    quaternion_table = scales.new_tensor([[.5, .5, .5, .5],
                                          [.5, -.5, -.5, -.5],
                                          [1., 0., 0., 0.]])
    return normal_axis, tangent_axes, quaternion_table[normal_axis]


class RTMaterialModel(nn.Module):
    """Official RT material/lighting parameters without duplicated base params.

    Official fresh-scene alpha and transmissivity defaults are .5. Continuation
    experiments may explicitly choose less disruptive values (e.g. .99/.95);
    such choices are persisted in config and are not official initialization.
    Post-baseline point count/order must remain fixed in the current port.
    """
    def __init__(self, base, config=None):
        super().__init__()
        from rt_port.vendor.encoding import SphMipEncoding

        self.config = dict(config or {})
        # Do not register the Ex model as a child: its existing optimizer owns it.
        object.__setattr__(self, "base", base)
        scales = base.get_scaling()
        if scales.ndim != 2 or scales.shape[-1] != 3:
            raise ValueError("RT port requires Ex4DGS three-axis ellipsoid scales")
        device, dtype = scales.device, scales.dtype
        n = len(scales)
        self.sph_dim = 16
        self.gsfeat_dim = 4
        self.XYZ = [int(v) for v in self.config.get("xyz_axis", [0, 1, 2])]
        if sorted(self.XYZ) != [0, 1, 2]:
            raise ValueError("xyz_axis must be a permutation of 0,1,2")
        self.shading_chunk_size = int(self.config.get("shading_chunk_size", 65536))
        if self.shading_chunk_size <= 0:
            raise ValueError("shading_chunk_size must be positive")
        self.checkpoint_shading = bool(self.config.get("checkpoint_shading", True))

        def logit_parameter(key, default):
            value = float(self.config.get(key, default))
            if not 0. < value < 1.:
                raise ValueError(f"{key} must lie strictly between 0 and 1")
            return nn.Parameter(torch.full((n, 1), math.log(value / (1 - value)),
                                           device=device, dtype=dtype))

        self._opacity = logit_parameter("initial_opacity", .5)
        self._transmissivity = logit_parameter("initial_transmissivity", .5)
        self._roughness = nn.Parameter(torch.zeros(n, 1, device=device, dtype=dtype))
        self._reflectance = nn.Parameter(torch.zeros(n, 1, device=device, dtype=dtype))
        self._language_feature = nn.Parameter(torch.zeros(n, 4, device=device, dtype=dtype))
        normal_axis, tangent_axes, permutation = initial_surfel_axes(scales)
        self.register_buffer("initial_normal_axis", normal_axis)
        self.register_buffer("tangent_axes", tangent_axes)
        self.register_buffer("surfel_frame_permutation", permutation)

        # Exact production architecture: nine mip levels, 512x1024x16 feature map.
        # Never silently shrink this network for smoke tests.
        self.dir_encoding = SphMipEncoding(9, 512, 16, 1, 1,
                                           bool(self.config.get("rand_init", False))).to(device)
        run_dim = 256
        self.light_mlp = nn.Sequential(
            nn.Linear(16 * 4 + 16, run_dim), nn.ReLU(inplace=True),
            nn.Linear(run_dim, run_dim), nn.ReLU(inplace=True),
            nn.Linear(run_dim, 4),
        ).to(device)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_transmissivity(self):
        return torch.sigmoid(self._transmissivity)

    @property
    def get_roughness(self):
        return torch.sigmoid(self._roughness)

    @property
    def get_reflectance(self):
        # Official reflectance offset is inverse_sigmoid(.5), exactly zero.
        return torch.sigmoid(self._reflectance)

    @property
    def get_language_feature(self):
        return torch.tanh(self._language_feature)

    @property
    def get_inside_mask(self):
        # The user requested the whole scene, not an assumed glass 3D sphere.
        return torch.ones_like(self._opacity, dtype=torch.bool)

    def occupancy_at(self, timestamp, training=False):
        occupancy = self.base.get_opacity_at_t(timestamp, training=training)
        if occupancy.shape != self._opacity.shape:
            raise RuntimeError("Ex4DGS topology changed; RT material rows no longer align")
        return occupancy

    def surfel_geometry(self, timestamp):
        scales = self.base.get_scaling()
        if len(scales) != len(self._opacity):
            raise RuntimeError("Ex4DGS topology changed; RT material rows no longer align")
        tangent_scales = scales.gather(1, self.tangent_axes).contiguous()
        base_rotation = F.normalize(self.base.get_rotation_at_t(timestamp), dim=-1)
        rotation = F.normalize(quaternion_product(base_rotation,
                                                  self.surfel_frame_permutation), dim=-1)
        return tangent_scales, rotation.contiguous()

    def optimizer_groups(self):
        """Only new params; geometry/color/occupancy use the restored Ex optimizer."""
        rates = {"opacity": .05, "transmissivity": .01, "roughness": .002,
                 "reflectance": .005, "feature": .002, "light_mlp": .0005,
                 "dir_encoding": .002}
        configured = self.config.get("learning_rates", {})
        unknown = set(configured) - set(rates)
        if unknown:
            raise ValueError(f"Unknown RT optimizer learning rates: {sorted(unknown)}")
        rates.update(configured)
        groups = [("opacity", [self._opacity]),
                  ("transmissivity", [self._transmissivity]),
                  ("roughness", [self._roughness]),
                  ("reflectance", [self._reflectance]),
                  ("feature", [self._language_feature]),
                  ("light_mlp", list(self.light_mlp.parameters())),
                  ("dir_encoding", list(self.dir_encoding.parameters()))]
        return [{"params": params, "lr": float(rates[name]), "name": name}
                for name, params in groups]

    def _shade_mlp(self, spec_feat, feature_map):
        # Official outer-product feature interaction, not a replacement SH light.
        wrap_input = (spec_feat[:, :, None] @ feature_map[:, None, :]).reshape(-1, 64)
        mlp_output = self.light_mlp(torch.cat((wrap_input, spec_feat), dim=-1))
        return torch.cat((torch.exp(mlp_output[:, :3] + math.log(.5)),
                          torch.sigmoid(mlp_output[:, 3:4])), dim=-1)

    def shade(self, reflected_directions, roughness, features):
        """Official shading math; chunk/checkpoint only the expensive light MLP."""
        from rt_port.vendor.sph_utils import cart2sph

        n = len(reflected_directions)
        if n == 0:
            return features.new_zeros((0, 3)), features.new_zeros((0, 1))
        features = F.normalize(features, dim=-1)
        angles = cart2sph(reflected_directions[:, self.XYZ])[:, 1:]
        uv = (angles / angles.new_tensor([[math.pi, 2 * math.pi]]))[:, [1, 0]]
        # SphMip mutates the newly created UV tensor as in the official code.
        spec_feat = self.dir_encoding(uv[None, :, None, :],
                                     roughness.reshape(-1, 1), index=0).reshape(-1, 16)
        outputs = []
        for offset in range(0, n, self.shading_chunk_size):
            stop = offset + self.shading_chunk_size
            arguments = (spec_feat[offset:stop], features[offset:stop])
            if self.checkpoint_shading and torch.is_grad_enabled():
                outputs.append(checkpoint(self._shade_mlp, *arguments, use_reentrant=False))
            else:
                outputs.append(self._shade_mlp(*arguments))
        result = torch.cat(outputs, dim=0)
        return result[:, :3], result[:, 3:4]

    def forward(self, camera, pipe, background, **kwargs):
        from rt_port.renderer import render
        return render(camera, self.base, self, pipe, background, **kwargs)
