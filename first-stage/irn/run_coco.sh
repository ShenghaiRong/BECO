#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python run_sample_coco.py \
--work_space ./test_coco/irn \
--train_cam_pass True  --make_cam_pass True --eval_cam_pass True 

CUDA_VISIBLE_DEVICES=0,1 python run_sample_coco.py \
--work_space ./test_coco/irn \
--cam_to_ir_label_pass True --train_irn_pass True \
--make_sem_seg_pass True --eval_sem_seg_pass True
