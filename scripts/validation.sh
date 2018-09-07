#!/usr/bin/env bash

cudapath='/usr/local/cuda/extras/CUPTI/lib64'
export LD_LIBRARY_PATH=$cudapath:$LD_LIBRARY_PATH

base_path=$(cd `dirname $0`/..; pwd)
addition_path=${base_path}'/RL_farmwork/dqn-prioritized-experience-replay'
export PYTHONPATH=${base_path}':'${addition_path}
echo ${PYTHONPATH}

logs_name='./convnet_val3'
mkdir -p ${logs_name}

export CUDA_VISIBLE_DEVICES=0

cond=("--optimizer=meta     --lr=2e-0 --batch_size=128 --meta_ckpt=./convnet_log/x_rnn_4"
      "--optimizer=momentum --lr=1e-3 --batch_size=128 --momentum=0.9"
      "--optimizer=adam     --lr=1e-3 --batch_size=128 --beta1=0.9 --beta2=0.999"
      "--optimizer=RMSprop  --lr=1e-3 --batch_size=128 --decay=0.99")

name=("x_rnn_4"
      "momentum"
      "adam"
      "RMSprop")

for ((i = 0; i < ${#cond[@]}; i++))
do
    chosen_name=${name[$i]}
    echo ${chosen_name}
    log_name=${logs_name}'/'${chosen_name}
    mkdir -p ${log_name}
    python -u train_val/use_meta_optimizer.py \
        --data_dir=./cifar10_data \
        --resnet_size=14 \
        --model_dir=${log_name} \
        --data_format=channels_last \
        --problem=convnet \
        ${cond[$i]} \
        2>&1 | tee ${log_name}'/logs.txt'
done
