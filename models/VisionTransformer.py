import torch
import torch.nn as nn
import timm


class VisionTransformer(nn.Module):

    def __init__(self, rgb_backbone_name='vit_base_patch8_224.dino', out_indices=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        kwargs = {'features_only': True if out_indices else False}

        if out_indices:
            kwargs.update({'out_indices': out_indices})

        layers_keep = 12

        ## RGB backbone
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, **kwargs)
        # ! Use only the first k blocks.
        self.rgb_backbone.blocks = torch.nn.Sequential(
            *self.rgb_backbone.blocks[:layers_keep])  # Remove Block(s) from 5 to 11.

    def forward(self, rgb, upsample=True, last_layer=False):
        upsample_shape = rgb.shape[-2:]

        x = self.rgb_backbone.patch_embed(rgb)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        # x = self.rgb_backbone.blocks(x)
        # every block output
        feature_list = []
        for i, block in enumerate(self.rgb_backbone.blocks):
            x = block(x)
            feature_list.append(x)

        # x = self.rgb_backbone.norm(x)
        feature_list = [self.rgb_backbone.norm(x)[:, 1:].permute(0, 2, 1).contiguous()
                        for x in feature_list]

        # feat = x[:, 1:].permute(0, 2, 1).view(1, -1, 28, 28)  # view(1, -1, 14, 14)

        x = torch.cat((feature_list[3], feature_list[7], feature_list[11]), dim=1)
        rgb_patch_upsample = torch.nn.functional.interpolate(x.view(1, -1, 28, 28), size=upsample_shape,
                                                             mode='bilinear', align_corners=False)
        if last_layer and not upsample:
            return feature_list[11].view(1, -1, 28, 28), rgb_patch_upsample
        return rgb_patch_upsample
