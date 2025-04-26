import argparse
import os
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.A2RD import DualAsTs_v2
from models.dataset import get_data_loader
from models.VisionTransformer import VisionTransformer
from processing.debug_vis import vis_inter_feat
from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score
def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)
def infer_A2(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path)
    dual_A2RD = DualAsTs_v2()
    Dual_AS_A2_3Dto2D_path = rf'{args.checkpoint_folder}/dual_asfm_{args.class_name}/DualAsfm_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
    dual_A2RD.load_state_dict(torch.load(Dual_AS_A2_3Dto2D_path))
    dual_A2RD.to(device)
    dual_A2RD.eval()
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l ** 2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u ** 2)
    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    for b_i, ((rgb, pc, depth), gt, label, rgb_path) in tqdm(enumerate(test_loader),
                                                      desc=f'Dual ASFM inference : {args.class_name} ', total=len(test_loader)):
        rgb, pc, depth = rgb.to(device), pc.to(device), depth.to(device)
        with torch.no_grad():
            pred_pcd_feat, gt_pcd_feat, pred_rgb_feat, gt_rgb_feat = dual_A2RD(pc,
                                                                                rgb)
            dims_3d = gt_pcd_feat.shape[1]
            dims_2d = gt_rgb_feat.shape[1]
            pred_rgb_feat = pred_rgb_feat.view(1, dims_2d, -1)  
            gt_rgb_feat = gt_rgb_feat.view(1, dims_2d, -1)
            vis_inter_feat(rgb, gt, pred_pcd_feat, gt_pcd_feat, pred_rgb_feat, gt_rgb_feat, 0, 0, b_i, args.class_name)
            rgb_mask = (pred_rgb_feat.squeeze().permute(1, 0).sum(axis=-1) == 0)
            xyz_mask = (gt_pcd_feat.squeeze().permute(1, 0).sum(axis=-1) == 0)
            cos_2d = (torch.nn.functional.normalize(pred_rgb_feat.squeeze().permute(1, 0), dim=1) -
                      torch.nn.functional.normalize(gt_rgb_feat.squeeze().permute(1, 0), dim=1)).pow(2).sum(1).sqrt()
            cos_2d[rgb_mask] = 0.
            cos_2d = cos_2d.reshape(224, 224)
            cos_3d = (torch.nn.functional.normalize(pred_pcd_feat.squeeze().permute(1, 0), dim=1) -
                      torch.nn.functional.normalize(gt_pcd_feat.squeeze().permute(1, 0), dim=1)).pow(2).sum(1).sqrt()
            cos_3d[xyz_mask] = 0.
            cos_3d = cos_3d.reshape(224, 224)
            cos_comb = cos_2d*cos_3d
            cos_comb = cos_comb.reshape(1, 1, 224, 224)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = cos_comb.reshape(224, 224)
            gts.append(gt.squeeze().cpu().detach().numpy())  
            predictions.append((cos_comb / (cos_comb[cos_comb != 0].mean())).cpu().detach().numpy())  
            image_labels.append(label)  
            pixel_labels.extend(gt.flatten().cpu().detach().numpy())  
            image_preds.append(
                (cos_comb / torch.sqrt(cos_comb[cos_comb != 0].mean())).cpu().detach().numpy().max())  
            pixel_preds.extend((cos_comb / torch.sqrt(cos_comb.mean())).flatten().cpu().detach().numpy())  
            if args.produce_qualitatives:
                defect_class_str = rgb_path[0].split('/')[-3]
                image_name_str = rgb_path[0].split('/')[-1]
                save_path = f'{args.qualitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig, axs = plt.subplots(2, 3, figsize=(7, 7))
                denormalize = transforms.Compose([
                    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                ])
                rgb = denormalize(rgb)
                os.path.join(save_path, image_name_str)
                axs[0, 0].imshow(rgb.squeeze().permute(1, 2, 0).cpu().detach().numpy())
                axs[0, 0].set_title('RGB')
                axs[0, 1].imshow(gt.squeeze().cpu().detach().numpy())
                axs[0, 1].set_title('Ground-truth')
                axs[0, 2].imshow(depth.squeeze().permute(1, 2, 0).mean(axis=-1).cpu().detach().numpy())
                axs[0, 2].set_title('Depth')
                axs[1, 0].imshow(cos_3d.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 0].set_title('3D CosSimilarity')
                axs[1, 1].imshow(cos_2d.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 1].set_title('2D CosSimilarity')
                axs[1, 2].imshow(cos_comb.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 2].set_title('Combined Cosine Similarity')
                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, image_name_str), dpi=256)
                if args.visualize_plot:
                    plt.show()
    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))
    result_file_name = f'{args.quantitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
    title_string = f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs'
    header_string = 'AUPRO & P-AUROC & I-AUROC'
    results_string = f'{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & {au_pros[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}'
    if not os.path.exists(args.quantitative_folder):
        os.makedirs(args.quantitative_folder)
    with open(result_file_name, "w") as markdown_file:
        markdown_file.write(title_string + '\n' + header_string + '\n' + results_string)
    print(title_string)
    print("AUPRO@ | P-AUROC | I-AUROC")
    print(
        f'  {au_pros[0]:.3f}   |   {pixel_rocauc:.3f} |   {image_rocauc:.3f}',
        end='\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make inference with A2RD.')
    parser.add_argument('--dataset_path', default='./mv3d_pro', 
                        type=str, help='Dataset path.')
    parser.add_argument('--class_name', default="cable_gland", type=str,
                        choices=["bagel_subset", "bagel",
                                 "cable_gland", "carrot", "cookie", "dowel", "foam", "peach","potato", "rope", "tire",
                                 'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear',
                                 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help='Category name.')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_A2_mvtec', type=str,
                        help='Path to the folder containing A2s checkpoints.')
    parser.add_argument('--qualitative_folder', default='./results/dual_qualitatives_mvtec', type=str,
                        help='Path to the folder in which to save the qualitatives.')
    parser.add_argument('--quantitative_folder', default='./results/quantitatives_mvtec', type=str,
                        help='Path to the folder in which to save the quantitatives.')
    parser.add_argument('--epochs_no', default=100, type=int,
                        help='Number of epochs to train the A2s.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch dimension. Usually 16 is around the max.')
    parser.add_argument('--visualize_plot', default=False, action='store_true',
                        help='Whether to show plot or not.')
    parser.add_argument('--produce_qualitatives', default=False, action='store_true',
                        help='Whether to produce qualitatives or not.')
    args = parser.parse_args()
    infer_A2(args)
