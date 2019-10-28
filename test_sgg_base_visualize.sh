#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/sgg_res101_joint_pretrained.yaml --inference --resume 99999 --algorithm sg_baseline --visualize --instance 10 --use_freq_prior
