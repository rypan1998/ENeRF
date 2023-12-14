#!/bin/bash
export workspace=/media/fhy/新加卷/rypan/ENeRF/workspace
# DTU Training
# python -m torch.distributed.launch --nproc_per_node=2 train_net.py --cfg_file configs/enerf/dtu_pretrain.yaml distributed True gpus 0,1
# LLFF Training
python -m torch.distributed.launch --nproc_per_node=2 train_net.py --cfg_file configs/enerf/llff_eval.yaml distributed True gpus 0,1
