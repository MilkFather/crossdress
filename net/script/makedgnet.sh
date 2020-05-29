#!/bin/bash
. script/testenv.sh
Test_Dress_Environment

source clothenv/bin/activate
python3 "DG-Net/make.py" --input_folder $1 --config "DG-Net/outputs/E0.5new_reid0.5_w30000/config.yaml" --checkpoint_gen "DG-Net/outputs/E0.5new_reid0.5_w30000/gen_00100000.pt" --checkpoint_id "DG-Net/outputs/E0.5new_reid0.5_w30000/id_00100000.pt" --make_phase $2
deactivate