#!/usr/bin/env bash

cudapath='/usr/local/cuda/extras/CUPTI/lib64'
export LD_LIBRARY_PATH=$cudapath:$LD_LIBRARY_PATH

base_path=$(cd `dirname $0`/..; pwd)
addition_path=${base_path}'/RL_farmwork/dqn-prioritized-experience-replay'
export PYTHONPATH=${base_path}':'${addition_path}
echo ${PYTHONPATH}

logs_name='./log11'
val_logs_name='./image_net_val'

mkdir -p ${logs_name}
mkdir -p ${val_logs_name}

cond=("--x_use=x --y_use=none --CG_iter=4"
      "--x_use=x --y_use=rnn --CG_iter=2"
      "--x_use=x --y_use=rnn --CG_iter=4"
      "--x_use=x --y_use=none --CG_iter=20")

name=("x_none" "x_rnn_2" "x_rnn_4" "full_CG")

cuda_devices=("0" "1" "2" "3")

for ((i = 0; i < ${#cond[@]}; i++))
do
    chosen_name=${name[$i]}
    echo ${chosen_name}

    log_name=${logs_name}'/'${chosen_name}
    mkdir -p ${log_name}

    val_log_name=${val_logs_name}'/'${chosen_name}
    mkdir -p ${val_log_name}

    export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
    python -u train_val/train_meta_optimizer.py \
        --data_dir=./cifar10_data \
        --batch_size=128 \
        --meta_roll_back=10 \
        --resnet_size=20 \
        --model_dir=${log_name} \
        --keep_prob=0.5 \
        --data_format=channels_last \
        --meta_lr=1e-2 \
        --epochs_per_eval=250 \
        --problem=resnet \
        ${cond[$i]} \
        2>&1 | tee ${log_name}'/logs.txt' && \
    python -u train_val/use_meta_resnet_on_imagenet.py \
        --data_dir=/home/data/dataset/ImageNet_2012 \
        --batch_size=64 \
        --resnet_size=18 \
        --model_dir=${val_log_name} \
        --data_format=channels_last \
        --problem=resnet \
        --optimizer=meta \
        --lr=0.5 \
        --meta_ckpt=${log_name} \
        ${cond[$i]} \
        2>&1 | tee ${val_log_name}'/logs.txt' &
done



