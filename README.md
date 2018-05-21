# Meta Learning with Hessian Free Approach in Neural Nets Training
This is a Tensorflow implementation of MLHF optimizer, a generic second-order  meta-optimizer, which includes:

* souce code of MLHF built on 2 layer LSTM.
* meta-traning code on cuda-convnet and resnet of cifar10.
* valuation code by traning cuda-convnet and resnet on cifar10 and ImageNet.

The training model code was modified from [Tensorflow/model](https://github.com/tensorflow/models), while experience replay code was modified from [evaldsurtans/dqn-prioritized-experience-replay](https://github.com/evaldsurtans/dqn-prioritized-experience-replay). see git submodule for more detail.

__Recommened Runtime Environment__:
* __python 3.6__
* __TensorFlow 1.7.0__

One should be emphasized that although this work is based on Tensorflow, it does not mean that the current Tensorflow framework has the full capacity to impliment all technique detail of MLHF, e.g. the gradient of $H$'s inplimentation require the `tf.gradents` cancalculated in `Defun` while it's graph was not the default graph and not allowed in current Tensorflow. So, we hack some core code of tensorflow, which would not guarantee the compatibility in different tensorflow version any more. This part of hackfull and evil code can be view in `CustomOp/gradients_impl.py`. 

## meta-training and validation on CUDA-convnet or ResNet of cifar10
First, prepare dataset and environment:
```bash
git clone --recurse-submodules -j8 git@github.com:ozzzp/MLHF.git
cd MLHF

python models/official/resnet/cifar10_download_and_extract.py --data_dir=./cifar10_data

base_path=$(pwd)
addition_path=${base_path}'/RL_farmwork/dqn-prioritized-experience-replay'
export PYTHONPATH=${base_path}':'${addition_path}

```
To meta-train on CUDA-convnet, run:
```bash
python train_val/train_meta_optimizer.py \
        --data_dir=./cifar10_data \
        --batch_size=64 \
        --meta_roll_back=10 \
        --model_dir=./cuda_convnet_log \
        --keep_prob=0.3 \
        --data_format=channels_last \
        --meta_lr=1e-3 \
        --train_epochs=250 \
        --problem=convnet \
        --x_use=x --y_use=rnn --CG_iter=4
```
To meta-train on ResNet, run:
```bash
python  train_val/train_meta_optimizer.py \
        --data_dir=./cifar10_data \
        --batch_size=128 \
        --meta_roll_back=10 \
        --resnet_size=20 \
        --model_dir=./resnet_log \
        --keep_prob=0.5 \
        --data_format=channels_last \
        --meta_lr=1e-2 \
        --epochs_per_eval=250 \
        --problem=resnet \
        --x_use=x --y_use=rnn --CG_iter=4
```
To evaluate by training on cifar10, run:
```bash
 python -u train_val/use_meta_optimizer.py \
        --data_dir=./cifar10_data \
        --batch_size=128 \
        --model_dir=./eval_cuda_convnet\
        --data_format=channels_last \
        --problem=convnet \
        --optimizer=meta \
        --train_epochs=250 \
        --lr=1 \
        --meta_ckpt=./cuda_convnet_log \
        --x_use=x --y_use=rnn --CG_iter=4
```
To evaluate by training imagenet on resnet, first, preapre ImageNet dataset as [here](https://github.com/tensorflow/models/tree/master/research/inception) to `./ImageNet_2012`, then, run:
```bash
 python --train_val/use_meta_resnet_on_imagenet.py \
        --data_dir=./ImageNet_2012 \
        --batch_size=64 \
        --resnet_size=18 \
        --model_dir=./eval_resnet \
        --data_format=channels_last \
        --problem=resnet \
        --optimizer=meta \
        --lr=1 \
        --meta_ckpt=./resnet_log \
        --x_use=x --y_use=rnn --CG_iter=4
```
Or, another choice is to modify and run scripts in `./scripts`

## extend to new model

The MLHF is a general optimizer, but we have impliment the minimal operators's difference forward and loss of experiment, this might be the main task that extend to new model. This part of code can be view in `CustomOp/op_r_forward.py` and `CustomOp/hession_loss.py`. also, To register new operator's type to RNN, view `CustomOp/rnn.py`.

## dicussion and feedback

Any discusstion, feedback or bugs report about MLHF are welcome. But it's not very recommend to contribute the application or extenction of MLHF (e.g. extend to new dataset, new model, more ops) to this repository, consider it's still a experiment project and might not be merged in time. _If you do such things or want to do, just fork this repository, and modify as your managed._   

