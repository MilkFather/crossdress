#!/bin/bash

Test_Pose_Environment() {
    if [ ! -d "poseenv" ]; then
        echo "Virtual environment for Pose-Transfer module not found. Creating one";
        virtualenv "poseenv"
        echo "Setting up virtual environment"
        source ./poseenv/bin/activate
        python3 -m pip install -r "Pose-Transfer/requirements.txt"
        deactivate
    fi
}

Test_Dress_Environment() {
    if [ ! -d "dressenv" ]; then
        echo "Virtual environment for DG-Net module not found. Creating one"
        virtualenv "dressenv"
        echo "Setting up virtual environment"
        source ./dressenv/bin/activate
        python3 -m pip install -r "DG-Net/requirements.txt"
        deactivate
    fi
}

Test_ABD_Environment() {
    if [ ! -d "abdenv" ]; then
        echo "Virtual environment for ABD-Net module not found. Creating one"
        virtualenv "abdenv"
        echo "Setting up virtual environment"
        source ./abdenv/bin/activate
        python3 -m pip install -r "ABD-Net/requirenemts.txt"
        deactivate
    fi
}