#!/bin/zsh

Test_Pose_Environment() {
    if [ ! -d "poseenv" ]; then
        echo "Virtual environment for Pose-Transfer module not found. Creating one";
        python3 -m venv "poseenv"
        echo "Setting up virtual environment"
        source ./poseenv/bin/activate
        python3 -m pip install -r "Pose-Transfer/requirements.txt" -i "https://mirrors.aliyun.com/pypi/simple/"
        deactivate
    fi
}

Test_Dress_Environment() {
    if [ ! -d "dressenv" ]; then
        echo "Virtual environment for DG-Net module not found. Creating one"
        python3 -m venv "dressenv"
        echo "Setting up virtual environment"
        source ./dressenv/bin/activate
        python3 -m pip install -r "DG-Net/requirements.txt" -i "https://mirrors.aliyun.com/pypi/simple/"
        deactivate
    fi
}