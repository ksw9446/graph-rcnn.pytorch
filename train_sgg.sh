#!/bin/bash

echo 'CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/sgg_res101_step.yaml --algorithm sg_grcnn'
echo ''
#python -m torch.distributed.launch --nproc_per_node=2 main.py --config-file configs/sgg_res101_joint.yaml --algorithm sg_grcnn # maybe 3 days...
#python -m torch.distributed.launch --nproc_per_node=2 main.py --config-file configs/sgg_res101_step.yaml --algorithm sg_grcnn
CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/sgg_res101_step.yaml --algorithm sg_grcnn
echo ''
echo 'CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/sgg_res101_step.yaml --algorithm sg_grcnn'

