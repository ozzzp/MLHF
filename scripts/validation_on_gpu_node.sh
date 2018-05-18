#!/usr/bin/env bash

cudapath='/usr/local/cuda/extras/CUPTI/lib64'
export LD_LIBRARY_PATH=$cudapath:$LD_LIBRARY_PATH

base_path=$(cd `dirname $0`/..; pwd)
addition_path=${base_path}'/RL_farmwork/dqn-prioritized-experience-replay'
export PYTHONPATH=${base_path}':'${addition_path}
echo ${PYTHONPATH}

logs_name='./image_net_val'
mkdir -p ${logs_name}

cond=("--optimizer=momentum --lr=0.025 --batch_size=64"
      "--optimizer=RMSprop --lr=0.0025 --batch_size=64"
      "--optimizer=adam --lr=0.0025 --batch_size=64"
      "--optimizer=SGD --lr=0.025 --batch_size=64")

name=("momentum" "RMSprop" "adam" "SGD")

cuda_devices=("0" "1" "2" "3")

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
