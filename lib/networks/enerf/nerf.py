import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(NeRF, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.lr0 = nn.Sequential(nn.Linear(8+16, hid_n),
                                 nn.ReLU()) # (8+16)*64
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ]) #? range 为 0，所以这是个空迭代，lrs 容器为空（为之后预留的？还是可以个性化修改？）
        # softplus(x) = log(1 + exp(x)). 把 x 非线性变换到 (0, +∞)
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
                nn.Linear(64+24+feat_ch+4, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, 1),
                nn.ReLU())
        self.lr0.apply(weights_init) # 权重初始化
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    def forward(self, vox_feat, img_feat_rgb_dir):
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1] # 返回除了最后一个维度（通道数）的维度
        img_feat = self.agg(img_feat_rgb_dir) # f_img
        S = img_feat_rgb_dir.shape[2]
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1) # 将 f_voxel 和 f_img 沿最后一个维度（-1）进行拼接，作为 MLP 的输入
        x = self.lr0(vox_img_feat) #? f_voxel=8, f_img=16
        for i in range(len(self.lrs)): #? lrs 是空容器，所以这里并没有执行
            x = self.lrs[i](x)
        sigma = self.sigma(x) # 密度 σ

        #? 下面得到颜色的过程挺抽象的
        x = torch.cat((x, vox_img_feat), dim=-1)
        x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1) # 将 x 的形状修改为 (B, -1, 1, x.shape[-1])，并进行重复复制 S 次
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x), dim=-2)
        color = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        return torch.cat([color, sigma], dim=-1)

class Agg(nn.Module):
    def __init__(self, feat_ch):
        """
        聚合：f_img = ψ(f1...fN)
        """
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if cfg.enerf.viewdir_agg:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]
        if cfg.enerf.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)

class MVSNeRF(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(MVSNeRF, self).__init__()
        self.hid_n = hid_n
        self.lr0 = nn.Sequential(nn.Linear(8+feat_ch*3, hid_n),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ])
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
                nn.Linear(hid_n, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, 3))
        self.lr0.apply(weights_init)
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    def forward(self, vox_feat, img_feat_rgb_dir):
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        # img_feat = self.agg(img_feat_rgb_dir)
        img_feat = torch.cat([img_feat_rgb_dir[..., i, :-4] for i in range(N_views)] , dim=-1)
        S = img_feat_rgb_dir.shape[2]
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)
        x = self.lr0(vox_img_feat)
        for i in range(len(self.lrs)):
            x = self.lrs[i](x)
        sigma = self.sigma(x)
        # x = torch.cat((x, vox_img_feat), dim=-1)
        # x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        # x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color = torch.sigmoid(self.color(x))
        return torch.cat([color, sigma], dim=-1)



def weights_init(m):
    '''
    初始化神经网络的权重。如果是线性层，则对其权重进行 Kaiming Normal 初始化，并把偏置项初始化为零。
    '''
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

