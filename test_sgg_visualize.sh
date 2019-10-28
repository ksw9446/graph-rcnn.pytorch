#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/sgg_res101_step.yaml --inference --resume 14999 --algorithm sg_grcnn --visualize --instance 10 #--use_freq_prior # 
echo 'CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/sgg_res101_step.yaml --inference --resume 13999 --algorithm sg_grcnn'
