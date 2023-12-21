import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .feature_net import FeatureNet
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from lib.config import cfg
from .nerf import NeRF

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.enerf.cas_config.num):
            if i == 0: # coarse 阶段
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))))
            else: # fine 阶段
                cost_reg_l = CostRegNet(int(32 * (2**(-i))))
            setattr(self, f'cost_reg_{i}', cost_reg_l) # self.cost_reg_0/1 = cost_reg_l
            nerf_l = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
            setattr(self, f'nerf_{i}', nerf_l)

    def render_rays(self, rays, **kwargs):
        '''
        处理光线渲染：采样、特征提取和 NeRF 模型生成输出
        '''
        level, batch, im_feat, feat_volume, nerf_model = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['nerf_model']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.enerf.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        # 如果图像数据在输入神经网络之前被缩放和归一化处理了，那么unpreprocess函数就可以将这些变换逆转回去，以便于图像显示或后续处理
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[level])
        up_feat_scale = cfg.enerf.cas_config.render_scale[level] / cfg.enerf.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.enerf.cas_config.render_scale[level]), int(W_O * cfg.enerf.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level) # B * N * S * (8+3+4)
        net_output = nerf_model(vox_feat, img_feat_rgb_dir)
        net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
        outputs = utils.raw2outputs(net_output, z_vals, cfg.enerf.white_bkgd)
        return outputs

    def batchify_rays(self, rays, **kwargs):
        '''
        批量处理光线：将光线分成小批量，并对每个批量调用 render_rays() 函数，然后将结果合并
        '''
        all_ret = {}
        chunk = cfg.enerf.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:, i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret


    def forward_feat(self, x):
        '''
        前向传播特征提取：通过 FeatureNet 提取输入数据 x 的特征，并返回不同级别的特征。
        '''
        B, S, C, H, W = x.shape # S 指序列长度
        x = x.view(B*S, C, H, W) # 将 B 和 S 合并，从而让 feature_net 能独立处理每个图像
        feat2, feat1, feat0 = self.feature_net(x)
        feats = { # 这里手动指定了新的分辨率，而不是使用 feat 的分辨率，只保留了 feat 的通道数
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)), # H*W*8
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)), # H/2*W/2*16
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)), # H/4*W/4*32
                }
        return feats

    def forward_render(self, ret, batch):
        '''
        前向传播渲染：似乎没有被调用
        '''
        B, _, _, H, W = batch['src_inps'].shape
        rgb = ret['rgb'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        rgb = self.cnn_renderer(rgb)
        ret['rgb'] = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)


    def forward(self, batch):
        '''
        前向传播：通过多个级别的特征提取和渲染流程，生成最终的输出。
        调用过程：当调用 self.network(batch) 时，实际上会调用 Network 类中的 __call__ 方法，而当前类继承自 torch.nn.Module 类，而 Module 类中的 __call__ 方法会调用模型类中定义的 forward 方法。
        '''
        feats = self.forward_feat(batch['src_inps'])
        ret = {}
        depth, std, near_far = None, None, None
        for i in range(cfg.enerf.cas_config.num):
            feature_volume, depth_values, near_far = utils.build_feature_volume(
                    feats[f'level_{i}'], # level_0&1 for cost volume, level_2 for NeRF
                    batch,
                    D=cfg.enerf.cas_config.volume_planes[i], # 深度平面数量
                    depth=depth,
                    std=std,
                    near_far=near_far,
                    level=i)
            feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
            depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
            # UV(2) +  ray_o (3) + ray_d (3) + ray_near_far (2) + volume_near_far (2)
            im_feat_level = cfg.enerf.cas_config.render_im_feat_level[i]
            ret_i = self.batchify_rays(
                    rays=rays,
                    feature_volume=feature_volume,
                    batch=batch,
                    im_feat=feats[f'level_{im_feat_level}'],
                    nerf_model=getattr(self, f'nerf_{i}'),
                    level=i)
            # if i == 1:
                # self.forward_render(ret_i, batch)
            if cfg.enerf.cas_config.depth_inv[i]:
                ret_i.update({'depth_mvs': 1./depth})
            else:
                ret_i.update({'depth_mvs': depth})
            ret_i.update({'std': std})
            if ret_i['rgb'].isnan().any():
                __import__('ipdb').set_trace()
            ret.update({key+f'_level{i}': ret_i[key] for key in ret_i})
        return ret
