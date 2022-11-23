#!/bin/sh

CUDA_VISIBLE_DEVICES=2 python run_sample.py \
--train_cam_pass True  --make_cam_pass True --eval_cam_pass True 

CUDA_VISIBLE_DEVICES=2,3 python run_sample.py \
--cam_to_ir_label_pass True --train_irn_pass True \
--make_sem_seg_pass True --eval_sem_seg_pass True
