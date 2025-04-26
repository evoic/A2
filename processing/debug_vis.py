import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import transforms
def normalize_xx(x):
    assert x.ndim == 1
    return (x - x.min()) / (x.max() - x.min())
def vis_inter_feat(rgb, gt_mask, pred_pcd_feat, gt_pcd_feat, pred_rgb_feat, gt_rgb_feat, epoch: int, batch_i: int,
                   i: int, c_name: str):
    plt_savepath = r"./vis"
    for_vis_pred_rgb = pred_rgb_feat.squeeze(dim=0).detach().cpu().reshape(-1, 224, 224)
    for_vis_gt_rgb = gt_rgb_feat.squeeze(dim=0).detach().cpu().reshape(-1, 224, 224)
    pred_unorganized_pcd_feat = pred_pcd_feat.squeeze(dim=0).detach().cpu().reshape(-1, 224 * 224).transpose(1, 0)
    gt_unorganized_pcd_feat = gt_pcd_feat.squeeze(dim=0).detach().cpu().reshape(-1, 224 * 224).transpose(1, 0)
    nonzero_indices = torch.nonzero(torch.all(gt_unorganized_pcd_feat != 0, dim=1))
    gt_unorganized_pc_no_zeros = gt_unorganized_pcd_feat[nonzero_indices, :].squeeze(dim=1)
    pred_unorganized_pc_no_zeros = pred_unorganized_pcd_feat[nonzero_indices, :].squeeze(dim=1)
    pred_pcd_feat_F_Length = torch.sqrt((pred_unorganized_pc_no_zeros ** 2).sum(1))
    pred_pcd_feat_norm_Length = ((pred_pcd_feat_F_Length - pred_pcd_feat_F_Length.min()) /
                                 (pred_pcd_feat_F_Length.max() - pred_pcd_feat_F_Length.min()))
    gt_pcd_feat_F_Length = torch.sqrt((gt_unorganized_pc_no_zeros ** 2).sum(1))
    gt_pcd_feat_norm_Length = ((gt_pcd_feat_F_Length - gt_pcd_feat_F_Length.min()) /
                               (gt_pcd_feat_F_Length.max() - gt_pcd_feat_F_Length.min()))
    for_vis_gt_pcd = torch.zeros((224, 224))
    for_vis_pred_pcd = torch.zeros((224, 224))
    for_vis_gt_pcd.view(-1)[nonzero_indices] = gt_pcd_feat_norm_Length.unsqueeze(-1)
    for_vis_pred_pcd.view(-1)[nonzero_indices] = pred_pcd_feat_norm_Length.unsqueeze(-1)
    rgb_mask = (for_vis_pred_rgb.reshape(-1, 224*224).permute(1, 0).sum(axis=-1) == 0)
    xyz_mask = (gt_unorganized_pcd_feat.sum(axis=-1) == 0)
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)
    cos_sim_3D = metric(pred_unorganized_pcd_feat[~xyz_mask], gt_unorganized_pcd_feat[~xyz_mask]).mean()
    cos_sim_2D = metric(for_vis_pred_rgb.reshape(-1, 224*224).permute(1, 0)[~rgb_mask],
                        for_vis_gt_rgb.reshape(-1, 224*224).permute(1, 0)[~rgb_mask]).mean()
    cos_3d = (torch.nn.functional.normalize(pred_unorganized_pcd_feat, dim=1) -
              torch.nn.functional.normalize(gt_unorganized_pcd_feat, dim=1)).pow(2).sum(1).sqrt()
    cos_3d[xyz_mask] = 0.
    cos_3d = cos_3d.reshape(224, 224)
    cos_2d = (torch.nn.functional.normalize(for_vis_pred_rgb.reshape(-1, 224 * 224).permute(1, 0), dim=1) -
              torch.nn.functional.normalize(for_vis_gt_rgb.reshape(-1, 224 * 224).permute(1, 0), dim=1)).pow(2).sum(
        1).sqrt()
    cos_2d[xyz_mask] = 0.
    cos_2d = cos_2d.reshape(224, 224)
    cos_comb = cos_2d * cos_3d
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    rgb = denormalize(rgb)
    if gt_mask is None:
        gt_mask = np.zeros((1, 1, 224, 224))
    hot_color_type = plt.cm.jet
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    axs[0, 0].set_title('rgb')
    axs[0, 0].imshow(rgb.squeeze().permute(1, 2, 0).cpu().detach().numpy())
    axs[0, 1].set_title('gt')
    axs[0, 1].imshow(gt_mask.squeeze().cpu().detach().numpy())
    axs[0, 2].set_title('3D*2D_CosSim')
    axs[0, 2].imshow(cos_comb.numpy(), cmap=hot_color_type)
    axs[1, 0].set_title('gt_pcd')
    axs[1, 0].imshow(for_vis_gt_pcd.numpy(), cmap=hot_color_type)
    axs[1, 1].set_title('pred_pcd')
    axs[1, 1].imshow(for_vis_pred_pcd.numpy(), cmap=hot_color_type)
    axs[1, 2].set_title(f'3D_CosSim_{cos_sim_3D:.4f}')
    axs[1, 2].imshow(cos_3d.numpy(), cmap=hot_color_type)
    _gt_2d = np.sqrt((for_vis_gt_rgb.numpy() ** 2).sum(0))
    _gt_2d.reshape(-1)[xyz_mask] = 0.
    axs[2, 0].set_title(f'gt_rgb_{for_vis_gt_rgb.shape[0]}')
    axs[2, 0].imshow(_gt_2d, cmap=hot_color_type)
    _pred_2d = np.sqrt((for_vis_pred_rgb.numpy() ** 2).sum(0))
    _pred_2d.reshape(-1)[xyz_mask] = 0.
    axs[2, 1].set_title(f'pred_rgb{for_vis_pred_rgb.shape[0]}')
    axs[2, 1].imshow(_pred_2d, cmap=hot_color_type)
    axs[2, 2].set_title(f'2D_CosSim_{cos_sim_2D:.4}')
    axs[2, 2].imshow(cos_2d.numpy(), cmap=hot_color_type)
    plt.tight_layout()
    plt.savefig(os.path.join(plt_savepath,
                             f"{c_name}-Epoch-{epoch}-batch-{batch_i}-i-{i}.png"), dpi=256)
    plt.close()
