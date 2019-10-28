#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/faster_rcnn_res101.yaml --inference # --visualize --instance 1

