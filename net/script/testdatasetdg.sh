#!/bin/bash

. script/testenv.sh
Test_Cloth_Environment

source clothenv/bin/activate
cd DG-Net
python3 train.py --config configs/latest.yaml
cd reid_eval
python3 test_2label.py --name E0.5new_reid0.5_w30000  --which_epoch 100000 --multi
cd ..
cd ..
deactivate