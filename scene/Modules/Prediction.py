import torch
from einops import repeat
from torch import nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from .BasicBlocks import ResidualMLP
from ..Common import GaussianParameters
from typing import Tuple
import numpy as np
class PredictionNetwork(nn.Module):
    def __init__(self, ref_feats_dim: int, res_feats_dim: int, derive_factor: int) -> None:
        """
        Network to Gaussian primitives.
        :param ref_feats_dim: dimension of reference features.
        :param res_feats_dim: dimension of residual features.
        :param derive_factor: number of coupled primitives derived from each anchor primitive.
        """
        super().__init__()

        self.derive_factor = derive_factor

        # init prediction networks
        mlp_in_dim = ref_feats_dim + res_feats_dim
        self.means_pred_mlp = ResidualMLP(in_dim=mlp_in_dim, internal_dim=ref_feats_dim, out_dim=3, num_res_layer=1)
        self.covariance_pred_mlp = ResidualMLP(in_dim=mlp_in_dim, internal_dim=ref_feats_dim, out_dim=7, num_res_layer=1)

        self.opacity_pred_mlp = ResidualMLP(in_dim=mlp_in_dim, internal_dim=ref_feats_dim, out_dim=1, num_res_layer=1, tail_activation=nn.Sigmoid())
        self.color_pred_mlp = ResidualMLP(in_dim=mlp_in_dim, internal_dim=ref_feats_dim, out_dim=3, num_res_layer=2, tail_activation=nn.Sigmoid())

        self.learnable_module_names = {'means_pred_mlp', 'covariance_pred_mlp', 'opacity_pred_mlp', 'color_pred_mlp'}

    def forward(self, means: torch.Tensor, scaling_factors: torch.Tensor, ref_feats: torch.Tensor, res_feats: torch.Tensor,
                rgb_factors: torch.Tensor) -> Tuple[GaussianParameters, torch.Tensor, torch.Tensor]:
        """
        Predict Gaussian primitives.
        :param means: means of anchor primitives, shape (N, 3).
        :param scaling_factors: scaling factors of anchor primitives, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :param rgb_factors: rgb factors of anchor primitives, shape (3, ).
        :return gaussian_primitives: predicted Gaussian primitives.
        :return pred_opacities: predicted opacities, shape (N * K, 1).
        """
        feats = repeat(ref_feats, 'n c -> (n k) c', k=self.derive_factor)
        feats = torch.cat([feats, res_feats], dim=-1)  # shape (N * K, C + G)

        means_offset = self.means_pred_mlp(feats)  # shape (N * K, 3)
        pred_covariance = self.covariance_pred_mlp(feats)  # shape (N * K, 7)

        pred_opacities = self.opacity_pred_mlp(feats)  # shape (N * K, 1)
        pred_colors = self.color_pred_mlp(feats)  # shape (N * K, 3)

        scaling_factors = repeat(scaling_factors, 'n c -> (n k) c', k=self.derive_factor)
        rgb_factors = repeat(rgb_factors, 'n c -> (n k) c', k=self.derive_factor)
        means = repeat(means, 'n c -> (n k) c', k=self.derive_factor)


        coupled_primitive_mask = (pred_opacities > 0.).view(-1)
        means_scaling_factor, scales_scaling_factor = torch.chunk(scaling_factors, dim=-1, chunks=2)  # shape (N * K, 3)

        gaussian_primitives = GaussianParameters(
            means=means + means_offset * means_scaling_factor,
            scales=torch.sigmoid(pred_covariance[:, :3] * scales_scaling_factor),
            rotations=F.normalize(pred_covariance[:, 3:]),
            opacities=pred_opacities,
            colors=pred_colors * rgb_factors,
        )

        return gaussian_primitives, pred_opacities, coupled_primitive_mask

    @torch.no_grad()
    def pred_gaussian_means(self, means: torch.Tensor, scaling_factors_before_exp: torch.Tensor, ref_feats: torch.Tensor, res_feats: torch.Tensor) -> torch.Tensor:
        """
        Predict means of Gaussian primitives, used in adaptive control.
        :param means: means of anchor primitives, shape (N, 3).
        :param scaling_factors_before_exp: scaling factors of anchor Gaussians before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :return means: means of predicted Gaussian primitives, shape (N * K, 3).
        """
        feats = repeat(ref_feats, 'n c -> (n k) c', k=self.derive_factor)
        feats = torch.cat([feats, res_feats], dim=-1)  # shape (N * K, C + G)
        means_offset = self.means_pred_mlp(feats)  # shape (N * K, 3)

        means = repeat(means, 'n c -> (n k) c', k=self.derive_factor)
        means_scaling_factor = torch.exp(scaling_factors_before_exp[:, :3])
        means_scaling_factor = repeat(means_scaling_factor, 'n c -> (n k) c', k=self.derive_factor)
        means = means + means_offset * means_scaling_factor

        return means
    @torch.no_grad()
    def pred_gaussian_opacities(self, means: torch.Tensor, scaling_factors_before_exp: torch.Tensor, ref_feats: torch.Tensor, res_feats: torch.Tensor) -> torch.Tensor:
        """
        Predict means of Gaussian primitives, used in adaptive control.
        :param means: means of anchor primitives, shape (N, 3).
        :param scaling_factors_before_exp: scaling factors of anchor Gaussians before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :return means: means of predicted Gaussian primitives, shape (N * K, 3).
        """
        feats = repeat(ref_feats, 'n c -> (n k) c', k=self.derive_factor)
        feats = torch.cat([feats, res_feats], dim=-1)  # shape (N * K, C + G)
        opacities = self.opacity_pred_mlp(feats)  # shape (N * K, 3)

        return opacities
    
    @torch.no_grad()
    def pred_gaussian_scales(self, means: torch.Tensor, scaling_factors_before_exp: torch.Tensor, ref_feats: torch.Tensor, res_feats: torch.Tensor) -> torch.Tensor:
        """
        Predict means of Gaussian primitives, used in adaptive control.
        :param means: means of anchor primitives, shape (N, 3).
        :param scaling_factors_before_exp: scaling factors of anchor Gaussians before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :return means: means of predicted Gaussian primitives, shape (N * K, 3).
        """
        feats = repeat(ref_feats, 'n c -> (n k) c', k=self.derive_factor)
        feats = torch.cat([feats, res_feats], dim=-1)  # shape (N * K, C + G)
        pred_covariance = self.covariance_pred_mlp(feats)  # shape (N * K, 7)

        scales = pred_covariance[:, :3]
        
        scales_scaling_factor = torch.exp(scaling_factors_before_exp[:, 3:])
        scales_scaling_factor = repeat(scales_scaling_factor, 'n c -> (n k) c', k=self.derive_factor)
        scales = torch.sigmoid(scales * scales_scaling_factor)

        return scales