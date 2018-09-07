#!/usr/bin/env bash

cudapath='/usr/local/cuda/extras/CUPTI/lib64'
export LD_LIBRARY_PATH=$cudapath:$LD_LIBRARY_PATH

base_path=$(cd `dirname $0`/..; pwd)
addition_path=${base_path}'/RL_farmwork/dqn-prioritized-experience-replay'
export PYTHONPATH=${base_path}':'${addition_path}
echo ${PYTHONPATH}

logs_name='./image_net_val'
mkdir -p ${logs_name}

cond=("--optimizer=meta     --batch_size=64  --lr=5e-1    --meta_ckpt=./convnet_log/x_rnn_4"
      "--optimizer=momentum --batch_size=64  --lr=0.025   --momentum=0.9"
      "--optimizer=RMSprop  --batch_size=64  --lr=0.00025 --decay=0.99"
      "--optimizer=adam     --batch_size=64  --lr=0.00025 --beta1=0.9 --beta2=0.999")

name=("x_rnn_4", "momentum" "RMSprop" "adam")

cuda_devices=("0" "1" "2")

for ((i = 0; i < ${#cond[@]}; i++))
do
    chosen_name=${name[$i]}
    echo ${chosen_name}
    log_name=${logs_name}'/'${chosen_name}
    mkdir -p ${log_name}
    export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
    python -u train_val/use_meta_resnet_on_imagenet.py \
        --data_dir=/home/data/dataset/ImageNet_2012 \
        --resnet_size=18 \
        --model_dir=${log_name} \
        --data_format=channels_last \
        --problem=resnet \
        ${cond[$i]} \
        2>&1 | tee ${log_name}'/logs.txt' &
done
