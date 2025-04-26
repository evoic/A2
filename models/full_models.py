import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath
from pointnet2_ops import pointnet2_utils
from models.PointTransformer import PointTransformer
class FeatureExtractors(torch.nn.Module):
    def __init__(self, device,
                 rgb_backbone_name='vit_base_patch8_224_dino.dino', out_indices=None,
                 group_size=128, num_group=1024):
        super().__init__()
        self.device = device
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        layers_keep = 12
        ## RGB backbone
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, **kwargs)
        # ! Use only the first k blocks.
        self.rgb_backbone.blocks = torch.nn.Sequential(
            *self.rgb_backbone.blocks[:layers_keep])  # Remove Block(s) from 5 to 11.
        ## XYZ backbone
        self.xyz_backbone = PointTransformer(group_size=group_size, num_group=num_group)
        self.xyz_backbone.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        # ! Use only the first k blocks.
        self.xyz_backbone.blocks.blocks = torch.nn.Sequential(
            *self.xyz_backbone.blocks.blocks[:layers_keep])  # Remove Block(s) from 5 to 11.
    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        x = self.rgb_backbone.blocks(x)
        x = self.rgb_backbone.norm(x)
        feat = x[:, 1:].permute(0, 2, 1).view(1, -1, 28, 28)  # view(1, -1, 14, 14)
        return feat
    def forward(self, rgb, xyz):
        rgb_features = self.forward_rgb_features(rgb)
        xyz_features, center, ori_idx, center_idx, pos = self.xyz_backbone(xyz)
        return rgb_features, xyz_features, center, ori_idx, center_idx
