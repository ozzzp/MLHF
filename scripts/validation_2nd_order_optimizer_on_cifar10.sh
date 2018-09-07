#!/usr/bin/env bash

cudapath='/usr/local/cuda/extras/CUPTI/lib64'
export LD_LIBRARY_PATH=$cudapath:$LD_LIBRARY_PATH

base_path=$(cd `dirname $0`/..; pwd)
addition_path=${base_path}'/RL_farmwork/dqn-prioritized-experience-replay'
export PYTHONPATH=${base_path}':'${addition_path}
echo ${PYTHONPATH}

logs_name='./cifar10_val'
mkdir -p ${logs_name}

export CUDA_VISIBLE_DEVICES=0

cond=("--optimizer=meta --lr=8e-0 --batch_size=512 --meta_ckpt=./convnet_log/x_rnn_4"
      "--optimizer=meta --lr=1e-2 --batch_size=512 --damping=5e-3              --x_use=x --y_use=none --d_use=none --CG_iter=20"
      "--optimizer=meta --lr=1e-3 --batch_size=512 --damping=1e-0 --decay=0.99 --x_use=x --y_use=none --d_use=none --CG_iter=20 --damping_type=LM_heuristics"
      "--optimizer=kfac --lr=1e-1 --batch_size=512 --damping=1e-0 --decay=0.99")

name=("x_rnn_4"
      "HF(Fixed)"
      "HF(LM)"
      "kfac")

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
