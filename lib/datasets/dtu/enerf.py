import numpy as np
import os
from glob import glob
from lib.utils.data_utils import load_K_Rt_from_P, read_cam_file
from lib.datasets import enerf_utils
from lib.config import cfg
import imageio
import tqdm
from multiprocessing import Pool
import copy
import cv2
import random
from lib.config import cfg
from lib.utils import data_utils
import torch
if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split'] # train or test
        if 'scene' in kwargs: # 只针对某个场景进行 fine-tuning
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas(kwargs['ann_file']) # 划分 dtu 数据集用于 train or test 分布的 txt 文件
        self.depth_ranges = [425., 905.] # ! 深度范围不同数据集是不同的

    def build_metas(self, ann_file):
        scenes = [line.strip() for line in open(ann_file).readlines()]
        dtu_pairs = torch.load('data/mvsnerf/pairs.th') # 视图对

        self.scene_infos = {}
        self.metas = []
        if len(self.scenes) != 0: # 只处理指定场景（fine-tuning 阶段）
            scenes = self.scenes

        for scene in scenes:
            scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []} # 内外参、深度图路径和图像路径
            for i in range(49):
                cam_path = os.path.join(self.data_root, 'Cameras/train/{:08d}_cam.txt'.format(i))
                ixt, ext, _ = data_utils.read_cam_file(cam_path) # 内外参
                ext[:3, 3] = ext[:3, 3] # 提取前三行的第四列，即平移向量，代表相机在世界坐标系中的位置
                ixt[:2] = ixt[:2] * 4 # 提取前两行，即 fx,fy,cx,cy，乘以4可能是为了调整分辨率
                dpt_path = os.path.join(self.data_root, 'Depths/{}/depth_map_{:04d}.pfm'.format(scene, i)) # 文件夹不带 train 的原始图
                img_path = os.path.join(self.data_root, 'Rectified/{}_train/rect_{:03d}_3_r5000.png'.format(scene, i+1))
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                scene_info['dpt_paths'].append(dpt_path)
                scene_info['img_paths'].append(img_path)

            if self.split == 'train' and len(self.scenes) != 1:
                train_ids = np.arange(49).tolist()
                test_ids = np.arange(49).tolist()
            elif self.split == 'train' and len(self.scenes) == 1:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_train']
            else:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_val']
            scene_info.update({'train_ids': train_ids, 'test_ids': test_ids})
            self.scene_infos[scene] = scene_info # 保存每个场景的数据集信息

            cam_points = np.array([np.linalg.inv(scene_info['exts'][i])[:3, 3] for i in train_ids])
            for tar_view in test_ids: # 为每个测试视图 tar_view 选择最相关的训练视图（距离最近）
                cam_point = np.linalg.inv(scene_info['exts'][tar_view])[:3, 3] # tar_view 的相机位置
                distance = np.linalg.norm(cam_points - cam_point[None], axis=-1) # tar_view 和所有训练视图相机位置的距离
                argsorts = distance.argsort() # 对距离进行排序，以找到最近的训练视图
                argsorts = argsorts[1:] if tar_view in train_ids else argsorts # 如果 tar_view 也在训练视图中，那么 argsorts[0] 一定是它自己，所以需要排除
                input_views_num = cfg.enerf.train_input_views[1] + 1 if self.split == 'train' else cfg.enerf.test_input_views
                src_views = [train_ids[i] for i in argsorts[:input_views_num]] # 选择一定数量的较近的训练视图作为源视图
                self.metas += [(scene, tar_view, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if random.random() < 0.1: # 有（较小）概率将 tar_view 添加到源视图中
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num) # 随机选择
        scene_info = self.scene_infos[scene]

        tar_img = np.array(imageio.imread(scene_info['img_paths'][tar_view])) / 255.
        H, W = tar_img.shape[:2]
        tar_ext, tar_ixt = scene_info['exts'][tar_view], scene_info['ixts'][tar_view]
        if self.split != 'train': # only used for evaluation
            tar_dpt = data_utils.read_pfm(scene_info['dpt_paths'][tar_view])[0].astype(np.float32)
            tar_dpt = cv2.resize(tar_dpt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            tar_dpt = tar_dpt[44:556, 80:720]
            tar_mask = (tar_dpt > 0.).astype(np.uint8)
        else: # train mode: 使用全一矩阵作为 depth 和 mask
            tar_dpt = np.ones_like(tar_img)
            tar_mask = np.ones_like(tar_img)

        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views) # 源视图的图像（经过处理）、相机参数

        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_dpt': tar_dpt,
                        'tar_mask': tar_mask})
        ret.update({'near_far': np.array(self.depth_ranges).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.enerf.cas_config.num):
            rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            s = cfg.enerf.cas_config.volume_scale[i]
            if self.split != 'train': # evaluation
                tar_dpt_i = cv2.resize(tar_dpt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                ret.update({f'tar_dpt_{i}': tar_dpt_i.astype(np.float32)})
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            ret['meta'].update({f'h_{i}': H, f'w_{i}': W})
        return ret

    def read_src(self, scene_info, src_views):
        inps, exts, ixts = [], [], []
        for src_view in src_views:
            inps.append((np.array(imageio.imread(scene_info['img_paths'][src_view])) / 255.) * 2. - 1.)
            exts.append(scene_info['exts'][src_view])
            ixts.append(scene_info['ixts'][src_view])
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)

    def __len__(self):
        return len(self.metas)

