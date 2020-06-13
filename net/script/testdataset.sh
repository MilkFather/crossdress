#!/bin/bash

. script/testenv.sh
Test_ABD_Environment

source abdenv/bin/activate
cd ABD-Net
python3 train.py -s market1501_ex -t market1501_ex \
    --root $1 \
    --market1501_extra $2 \
    --flip-eval --eval-freq 1 \
    --label-smooth \
    --criterion xentlsro \
    --lambda-htri 0.1  \
    --data-augment crop random-erase \
    --margin 1.2 \
    --train-batch-size 64 \
    --height 128 \
    --width 64 \
    --optim adam --lr 0.0003 \
    --stepsize 20 40 \
    --gpu-devices 0,1,2,3 \
    --max-epoch 80 \
    --save-dir model \
    --arch resnet50 \
    --use-of \
    --branches abd \
    --abd-dan cam pam \
    --abd-np 2 \
    --shallow-cam \
    --use-ow
cd ..
deactivate