#!/bin/bash
. script/testenv.sh
Test_Dress_Environment

source dressenv/bin/activate
cd DG-Net
python3 "DG-Net/make.py" --input_folder "../$1" --make_phase $2
cd ..
deactivate