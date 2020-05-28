#!/bin/zsh
. script/testenv.sh
Test_Pose_Environment

source poseenv/bin/activate
python3 "Pose-Transfer/make.py" --dataroot $1 --name market_PATN --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 2 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir $2 --which_epoch latest --display_id 0 --make_phase $3
deactivate