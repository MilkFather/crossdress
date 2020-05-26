#!/bin/zsh

. script/testenv.sh

Test_Pose_Environment

source poseenv/bin/activate
python3 "Pose-Transfer/tool/generate_pose_map_market.py" --dataroot "../market-1501"
deactivate