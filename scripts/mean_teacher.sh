#!/bin/sh

PATH=`pwd`

CUDA_VISIBLE_DEVICES=0 ~/tf-2.3/bin/python3 $PATH/code/mean_teacher.py \
    --dataset TrainingV2\
    --epochs 100\
    --batch_size 32\
    --shuffle True\
    --num_parallel_calls 12\
    --buffer_size 1000000\
    --prefetch 1\
    --learning_rate 0.001\
    --beta_1 0.9\
    --beta_2 0.999\
    --resnet_ver 3\
    --resnet_n 1\
    --val_iteration 437\
    --lambda_u 0.1\
    --rampup_length 16\
    --ema_decay 0.999\
    --pr_curve_file True\

