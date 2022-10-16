#!/bin/sh

PATH=`pwd`

CUDA_VISIBLE_DEVICES=0 ~/tf-2.3/bin/python3 $PATH/code/VAT.py \
    --dataset TrainingV2\
    --epochs 100\
    --batch_size 32\
    --unlabeled_batch_size 32\
    --val_iterations 437\
    --shuffle True\
    --num_parallel_calls 12\
    --buffer_size 1000000\
    --prefetch 1\
    --learning_rate 0.001\
    --beta_1 0.9\
    --beta_2 0.999\
    --epsilon 0.001\
    --epsilon_adv 0.1\
    --lambda_u 0.01\
    --rampup_length 16\
    --resnet_ver 3\
    --resnet_n 1\
    --pr_curve_file True\

