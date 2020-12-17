#!/bin/bash

. script/testenv.sh
Test_Dress_Environment

source dressenv/bin/activate
cd DG-Net
python3 train.py --config configs/latest.yaml --gpu_ids 0,1,2,3
cd reid_eval
python3 test_2label.py --name latest  --which_epoch 100000 --multi
cd ..
cd ..
deactivate