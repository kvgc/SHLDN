#!/bin/sh

PATH=`pwd`

CUDA_VISIBLE_DEVICES=1 ~/tf-2.3/bin/python3 $PATH/code/WGAN-GP.py \
    --epochs 10\
    --iterations 200000\
    --batch_size 16\
    --z_dim 128\
    --n_critic 1\
    --LAMBDA 10\
    --shuffle True\
    --num_parallel_calls 12\
    --buffer_size 10000\
    --prefetch 1000\
    --G_learning_rate 0.0001\
    --G_beta_1 0\
    --G_beta_2 0.9\
    --D_learning_rate 0.0001\
    --D_beta_1 0\
    --D_beta_2 0.9\



