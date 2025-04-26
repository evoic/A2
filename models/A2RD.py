import timm
import torch
from torch import nn
from models.PointTransformer import PointTransformer, TransformerEncoder
from models.VisionTransformer import VisionTransformer
from models.feature_transfer_nets import FeatureProjectionMLP
from utils.pointnet2_utils import interpolating_points
group_size = 128
num_group = 1024
class AsTs(nn.Module):
    def __init__(self):
        super(AsTs, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher_token_dims = 384
        student_token_dims = 768
        self.teacher = PointTransformer(group_size, num_group, teacher_token_dims)
        self.teacher.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        layers_keep = 12
        self.teacher.blocks.blocks = torch.nn.Sequential(*self.teacher.blocks.blocks[:layers_keep])
        self.patch_embed = nn.Sequential(nn.Conv2d(3 * teacher_token_dims, student_token_dims,
                                                   kernel_size=8, stride=8, bias=True),
                                         nn.Identity())
        self.t2s = FeatureProjectionMLP(in_features=3 * teacher_token_dims, out_features=student_token_dims)
        self.student = Student_Point_T(token_dims=student_token_dims)
        self.student_to_vit = nn.Identity()
        self.froze_teacher()
    def forward(self, pc):
        unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)
        with torch.no_grad():
            teacher_features, center, ori_idx, center_idx, pos = self.teacher(unorganized_pc_no_zeros.contiguous())
            full_teacher_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                         center.permute(0, 2, 1),  
                                                         teacher_features)
        xyz_patch_full = torch.zeros((1, full_teacher_features.shape[1], 224 * 224),
                                     dtype=full_teacher_features.dtype, device=self.device)
        xyz_patch_full[..., nonzero_indices] = full_teacher_features
        t2s_feat = self.patch_embed(xyz_patch_full.reshape(1, 1152, 224, 224))
        t2s_tokens = t2s_feat.reshape(1, t2s_feat.shape[1], t2s_feat.shape[2] * t2s_feat.shape[3]).transpose(2, 1)
        student_features = self.student(t2s_tokens, center)
        s2v_1 = self.student_to_vit(student_features.transpose(-1, -2)[:, :, 0:768])
        s2v_2 = self.student_to_vit(student_features.transpose(-1, -2)[:, :, 768:768 * 2])
        s2v_3 = self.student_to_vit(student_features.transpose(-1, -2)[:, :, 768 * 2:768 * 3:])
        s_f = torch.cat((s2v_1, s2v_2, s2v_3), dim=-1).transpose(-1, -2)
        xyz_patch_full = torch.nn.functional.interpolate(s_f.view(1, -1, 28, 28), size=(224, 224),
                                                         mode='bilinear', align_corners=False)
        return s_f, nonzero_indices, xyz_patch_full  
    def froze_teacher(self):
        for name, param in self.teacher.named_parameters():
            param.requires_grad = False
class DualAsTs(nn.Module):
    def __init__(self):
        super(DualAsTs, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token_3d_dims = 384
        token_2d_dims = 768
        self.teacher_3d = PointTransformer(group_size, num_group, token_3d_dims)
        self.teacher_3d.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        layers_keep = 12
        self.teacher_3d.blocks.blocks = torch.nn.Sequential(*self.teacher_3d.blocks.blocks[:layers_keep])
        self.teacher_3d_resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.teacher_3d_average = torch.nn.AvgPool2d(kernel_size=3, stride=1)
        self.student_3d_resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.student_3d_average = torch.nn.AvgPool2d(kernel_size=3, stride=1)
        self.neck_3dto2d = nn.Sequential(
            nn.Conv2d(3 * token_3d_dims, token_2d_dims, kernel_size=8, stride=8, bias=True),
            nn.Identity())
        self.teacher_2d = VisionTransformer()
        self.neck_2dto3d = FeatureProjectionMLP(in_features= token_2d_dims, out_features=token_3d_dims)
        self.student_2d = Student_Point_T(token_dims=token_2d_dims, token_num=28 * 28)
        self.student_3d = Student_Point_T(token_dims=token_3d_dims, token_num=1024, depth=4)
        self.student_to_vit = nn.Identity()
        self.froze_teacher()
    def forward(self, pc, rgb):
        self.teacher_3d.eval()
        self.teacher_2d.eval()
        unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)
        with torch.no_grad():
            pcd_teacher_features, center, ori_idx, center_idx, pos = self.teacher_3d(
                unorganized_pc_no_zeros.contiguous())
            full_pcd_teacher_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                             center.permute(0, 2, 1),  
                                                             pcd_teacher_features)
            full_img_teacher_features = self.teacher_2d(rgb)  
        xyz_patch_full = torch.zeros((1, full_pcd_teacher_features.shape[1], 224 * 224),
                                     dtype=full_pcd_teacher_features.dtype, device=self.device)
        xyz_patch_full[..., nonzero_indices] = full_pcd_teacher_features
        xyz_patch_full_2d = xyz_patch_full.view(1, full_pcd_teacher_features.shape[1], 224, 224)
        xyz_patch_full_resized = self.teacher_3d_resize(self.teacher_3d_average(xyz_patch_full_2d))
        t2s_3dto2d_feat = self.neck_3dto2d(xyz_patch_full_resized)
        t2s_3dto2d_tokens = t2s_3dto2d_feat.reshape(1, t2s_3dto2d_feat.shape[1],
                                                    t2s_3dto2d_feat.shape[2] * t2s_3dto2d_feat.shape[3]).transpose(2, 1)
        student_2d_features = self.student_2d(t2s_3dto2d_tokens, center)
        s2v_1 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 0:768])
        s2v_2 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768:768 * 2])
        s2v_3 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768 * 2:768 * 3:])
        s_2d_f = torch.cat((s2v_1, s2v_2, s2v_3), dim=-1).transpose(-1, -2)
        pred_rgb_patch_full = torch.nn.functional.interpolate(s_2d_f.view(1, -1, 28, 28), size=(224, 224),
                                                              mode='bilinear', align_corners=False)
        center_2d_feature = full_img_teacher_features.view(1, -1, 224 * 224)[:, 768*2:768*3,:][..., center_idx.long()[0]]
        t2s_2dto3d_feat = self.neck_2dto3d(center_2d_feature.transpose(-1, -2).contiguous())
        t2s_2dto3d_token = t2s_2dto3d_feat
        student_3d_features = self.student_3d(t2s_2dto3d_token, center)
        full_pcd_student_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                         center.permute(0, 2, 1),  
                                                         student_3d_features)
        pred_xyz_patch_full = torch.zeros((1, full_pcd_teacher_features.shape[1], 224 * 224),
                                          dtype=full_pcd_student_features.dtype, device=self.device)
        pred_xyz_patch_full[..., nonzero_indices] = full_pcd_student_features
        pred_xyz_patch_full_2d = pred_xyz_patch_full.view(1, full_pcd_student_features.shape[1], 224, 224)
        pred_xyz_patch_full_resized = self.student_3d_resize(self.student_3d_average(pred_xyz_patch_full_2d))
        return (pred_xyz_patch_full_resized.reshape(1, -1, 224*224), xyz_patch_full_resized.reshape(1, -1, 224*224),
                pred_rgb_patch_full, full_img_teacher_features)
    def froze_teacher(self):
        for name, param in self.teacher_3d.named_parameters():
            param.requires_grad = False
        for name, param in self.teacher_2d.named_parameters():
            param.requires_grad = False
class DualAsTs_v2(nn.Module):
    def __init__(self):
        super(DualAsTs_v2, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token_3d_dims = 384
        token_2d_dims = 768
        self.teacher_3d = PointTransformer(group_size, num_group, token_3d_dims)
        self.teacher_3d.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        layers_keep = 12
        self.teacher_3d.blocks.blocks = torch.nn.Sequential(*self.teacher_3d.blocks.blocks[:layers_keep])
        self.teacher_3d_resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.teacher_3d_average = torch.nn.AvgPool2d(kernel_size=3, stride=1)
        self.student_3d_resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.student_3d_average = torch.nn.AvgPool2d(kernel_size=3, stride=1)
        self.neck_3dto2d = nn.Sequential(
            nn.Conv2d(3 * token_3d_dims, token_2d_dims, kernel_size=8, stride=8, bias=True),
            nn.Identity())
        self.teacher_2d = VisionTransformer()
        self.neck_2dto3d = nn.Sequential(torch.nn.Linear(768, (256+768)//2),
                                         torch.nn.GELU(),
                                         torch.nn.Linear((256+768)//2, 256),
                                         torch.nn.GELU(),
                                         torch.nn.Linear(256, 768//2),
                                         )
        self.student_2d = Student_Point_T(token_dims=token_2d_dims, token_num=28 * 28)
        self.student_3d = Student_Point_T(token_dims=token_3d_dims, token_num=28*28, depth=12)
        self.student_to_vit = nn.Identity()
        self.froze_teacher()
    def forward(self, pc, rgb):
        self.teacher_3d.train()
        self.teacher_2d.eval()
        unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)
        with torch.no_grad():
            pcd_teacher_features, center, ori_idx, center_idx, pos = self.teacher_3d(
                unorganized_pc_no_zeros.contiguous())
            full_pcd_teacher_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                             center.permute(0, 2, 1),  
                                                             pcd_teacher_features)
            vit_feat_28x28, full_img_teacher_features = self.teacher_2d(rgb, upsample=False, last_layer=True)  
        xyz_patch_full = torch.zeros((1, full_pcd_teacher_features.shape[1], 224 * 224),
                                     dtype=full_pcd_teacher_features.dtype, device=self.device)
        xyz_patch_full[..., nonzero_indices] = full_pcd_teacher_features
        xyz_patch_full_2d = xyz_patch_full.view(1, full_pcd_teacher_features.shape[1], 224, 224)
        xyz_patch_full_resized = self.teacher_3d_resize(self.teacher_3d_average(xyz_patch_full_2d))
        t2s_3dto2d_feat = self.neck_3dto2d(xyz_patch_full_resized)
        t2s_3dto2d_tokens = t2s_3dto2d_feat.reshape(1, t2s_3dto2d_feat.shape[1],
                                                    t2s_3dto2d_feat.shape[2] * t2s_3dto2d_feat.shape[3]).transpose(2, 1)
        student_2d_features = self.student_2d(t2s_3dto2d_tokens, center)
        s2v_1 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 0:768])
        s2v_2 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768:768 * 2])
        s2v_3 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768 * 2:768 * 3:])
        s_2d_f = torch.cat((s2v_1, s2v_2, s2v_3), dim=-1).transpose(-1, -2)
        pred_rgb_patch_full = torch.nn.functional.interpolate(s_2d_f.view(1, -1, 28, 28), size=(224, 224),
                                                              mode='bilinear', align_corners=False)
        t2s_2dto3d_feat = self.neck_2dto3d(vit_feat_28x28.view(1, -1, 28*28).transpose(-1, -2).contiguous())
        t2s_2dto3d_token = t2s_2dto3d_feat
        student_3d_features = self.student_3d(t2s_2dto3d_token, center)
        pred_xyz_patch_full_resized = torch.nn.functional.interpolate(student_3d_features.view(1, -1, 28, 28), size=(224, 224),
                                                              mode='bilinear', align_corners=False)
        return (pred_xyz_patch_full_resized.reshape(1, -1, 224*224), xyz_patch_full_resized.reshape(1, -1, 224*224),
                pred_rgb_patch_full, full_img_teacher_features)
    def froze_teacher(self):
        for name, param in self.teacher_3d.named_parameters():
            param.requires_grad = False
        for name, param in self.teacher_2d.named_parameters():
            param.requires_grad = False
class DualAsTs_mlp(nn.Module):
    def __init__(self):
        super(DualAsTs_mlp, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token_3d_dims = 384
        token_2d_dims = 768
        self.teacher_3d = PointTransformer(group_size, num_group, token_3d_dims)
        self.teacher_3d.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        layers_keep = 12
        self.teacher_3d.blocks.blocks = torch.nn.Sequential(*self.teacher_3d.blocks.blocks[:layers_keep])
        self.teacher_3d_resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.teacher_3d_average = torch.nn.AvgPool2d(kernel_size=3, stride=1)
        self.neck_3dto2d = nn.Sequential(
            nn.Conv2d(3 * token_3d_dims, token_2d_dims, kernel_size=8, stride=8, bias=True),
            nn.Identity())
        self.teacher_2d = VisionTransformer()
        self.neck_2dto3d = FeatureProjectionMLP(in_features=3 * token_2d_dims, out_features=token_3d_dims)
        self.student_2d = Student_Point_T(token_dims=token_2d_dims, token_num=28 * 28)
        self.student_3d = FeatureProjectionMLP(in_features=token_2d_dims, out_features=token_3d_dims*3)
        self.student_to_vit = nn.Identity()
        self.froze_teacher()
    def forward(self, pc, rgb):
        self.teacher_3d.eval()
        self.teacher_2d.eval()
        unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)
        with torch.no_grad():
            pcd_teacher_features, center, ori_idx, center_idx, pos = self.teacher_3d(
                unorganized_pc_no_zeros.contiguous())
            full_pcd_teacher_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                             center.permute(0, 2, 1),  
                                                             pcd_teacher_features)
            full_img_teacher_features = self.teacher_2d(rgb)  
        xyz_patch_full = torch.zeros((1, full_pcd_teacher_features.shape[1], 224 * 224),
                                     dtype=full_pcd_teacher_features.dtype, device=self.device)
        xyz_patch_full[..., nonzero_indices] = full_pcd_teacher_features
        xyz_patch_full_2d = xyz_patch_full.view(1, full_pcd_teacher_features.shape[1], 224, 224)
        xyz_patch_full_resized = self.teacher_3d_resize(self.teacher_3d_average(xyz_patch_full_2d))
        t2s_3dto2d_feat = self.neck_3dto2d(xyz_patch_full_resized)
        t2s_3dto2d_tokens = t2s_3dto2d_feat.reshape(1, t2s_3dto2d_feat.shape[1],
                                                    t2s_3dto2d_feat.shape[2] * t2s_3dto2d_feat.shape[3]).transpose(2, 1)
        student_2d_features = self.student_2d(t2s_3dto2d_tokens, center)
        s2v_1 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 0:768])
        s2v_2 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768:768 * 2])
        s2v_3 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768 * 2:768 * 3:])
        s_2d_f = torch.cat((s2v_1, s2v_2, s2v_3), dim=-1).transpose(-1, -2)
        pred_rgb_patch_full = torch.nn.functional.interpolate(s_2d_f.view(1, -1, 28, 28), size=(224, 224),
                                                              mode='bilinear', align_corners=False)
        student_3d_features = self.student_3d(full_img_teacher_features[:, 768*2:768*3, :, :].reshape(1, -1, 224*224).transpose(-1, -2),)
        return (student_3d_features.transpose(-1, -2), xyz_patch_full_resized.reshape(1, -1, 224*224),
                pred_rgb_patch_full, full_img_teacher_features)
    def froze_teacher(self):
        for name, param in self.teacher_3d.named_parameters():
            param.requires_grad = False
        for name, param in self.teacher_2d.named_parameters():
            param.requires_grad = False
class DualAsTs_bp(nn.Module):
    def __init__(self):
        super(DualAsTs, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token_3d_dims = 384
        token_2d_dims = 768
        self.teacher_3d = PointTransformer(group_size, num_group, token_3d_dims)
        self.teacher_3d.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        layers_keep = 12
        self.teacher_3d.blocks.blocks = torch.nn.Sequential(*self.teacher_3d.blocks.blocks[:layers_keep])
        self.neck_3dto2d = nn.Sequential(
            nn.Conv2d(3 * token_3d_dims, token_2d_dims, kernel_size=8, stride=8, bias=True),
            nn.Identity())
        self.teacher_2d = VisionTransformer()
        self.neck_2dto3d = FeatureProjectionMLP(in_features=3 * token_2d_dims, out_features=token_3d_dims)
        self.student_2d = Student_Point_T(token_dims=token_2d_dims, token_num=28 * 28)
        self.student_3d = Student_Point_T(token_dims=token_3d_dims, token_num=1024)
        self.student_to_vit = nn.Identity()
        self.froze_teacher()
    def forward(self, pc, rgb):
        unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)
        with torch.no_grad():
            pcd_teacher_features, center, ori_idx, center_idx, pos = self.teacher_3d(
                unorganized_pc_no_zeros.contiguous())
            full_pcd_teacher_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                             center.permute(0, 2, 1),  
                                                             pcd_teacher_features)
            full_img_teacher_features = self.teacher_2d(rgb)  
        xyz_patch_full = torch.zeros((1, full_pcd_teacher_features.shape[1], 224 * 224),
                                     dtype=full_pcd_teacher_features.dtype, device=self.device)
        xyz_patch_full[..., nonzero_indices] = full_pcd_teacher_features
        t2s_3dto2d_feat = self.neck_3dto2d(xyz_patch_full.reshape(1, 1152, 224, 224))
        t2s_3dto2d_tokens = t2s_3dto2d_feat.reshape(1, t2s_3dto2d_feat.shape[1],
                                                    t2s_3dto2d_feat.shape[2] * t2s_3dto2d_feat.shape[3]).transpose(2, 1)
        student_2d_features = self.student_2d(t2s_3dto2d_tokens, center)
        s2v_1 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 0:768])
        s2v_2 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768:768 * 2])
        s2v_3 = self.student_to_vit(student_2d_features.transpose(-1, -2)[:, :, 768 * 2:768 * 3:])
        s_2d_f = torch.cat((s2v_1, s2v_2, s2v_3), dim=-1).transpose(-1, -2)
        pred_rgb_patch_full = torch.nn.functional.interpolate(s_2d_f.view(1, -1, 28, 28), size=(224, 224),
                                                              mode='bilinear', align_corners=False)
        center_2d_feature = full_img_teacher_features.view(1, -1, 224 * 224)[..., center_idx.long()[0]]
        t2s_2dto3d_feat = self.neck_2dto3d(center_2d_feature.transpose(-1, -2).contiguous())
        t2s_2dto3d_token = t2s_2dto3d_feat
        student_3d_features = self.student_3d(t2s_2dto3d_token, center)
        full_pcd_student_features = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                         center.permute(0, 2, 1),  
                                                         student_3d_features)
        pred_xyz_patch_full = torch.zeros((1, full_pcd_teacher_features.shape[1], 224 * 224),
                                          dtype=full_pcd_student_features.dtype, device=self.device)
        pred_xyz_patch_full[..., nonzero_indices] = full_pcd_student_features
        return student_3d_features, pcd_teacher_features, pred_rgb_patch_full, full_img_teacher_features
    def froze_teacher(self):
        for name, param in self.teacher_3d.named_parameters():
            param.requires_grad = False
        for name, param in self.teacher_2d.named_parameters():
            param.requires_grad = False
class Student_Point_T(nn.Module):
    def __init__(self, token_dims=384, token_num=28 * 28, depth=12):
        super(Student_Point_T, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trans_dim = token_dims
        self.depth = depth
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.pos_embed_1d = nn.Parameter(torch.randn(1, token_num, self.trans_dim) * .02)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)
    def forward(self, teacher_features, center):
        pos = self.pos_embed(center)
        pos_zero = torch.zeros_like(pos).float().to(self.device)  
        feature_list = self.blocks(teacher_features, self.pos_embed_1d)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        if len(feature_list) == 12:
            x = torch.cat((feature_list[3], feature_list[7], feature_list[11]), dim=1)
        elif len(feature_list) == 8:
            x = torch.cat((feature_list[1], feature_list[4], feature_list[7]), dim=1)
        elif len(feature_list) == 4:
            x = torch.cat((feature_list[1], feature_list[2], feature_list[3]), dim=1)
        else:
            x = feature_list[-1]
        return x
class Student_Vit(nn.Module):
    def __init__(self, rgb_backbone_name='vit_base_patch8_224_dino.dino', out_indices=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        layers_keep = 12
        rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, **kwargs)
        self.pos_embed = nn.Parameter(torch.randn(rgb_backbone.pos_embed.size()) * .02)
        self.blocks = torch.nn.Sequential(
            *self.rgb_backbone.blocks[:layers_keep])  
        self.norm_pre = nn.Identity()  
    def forward(self, teacher_features):
        x = teacher_features
        x = self.rgb_backbone._pos_embed(x)
        x = self.norm_pre(x)
        feature_list = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_list.append(x)
        feature_list = [self.rgb_backbone.norm(x)[:, 1:].permute(0, 2, 1).contiguous()
                        for x in feature_list]
        x = torch.cat((feature_list[3], feature_list[7], feature_list[11]), dim=1)
        rgb_patch_upsample = None
        return rgb_patch_upsample
