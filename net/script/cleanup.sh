#!/bin/bash

DeleteGeneratedDataset () {
    dataroot=$1
    if [ -d "$dataroot/bounding_box_trainK" ]; then
        rm -rf "$dataroot/bounding_box_trainK" 
    fi
    if [ -d "$dataroot/bounding_box_train_pose" ]; then
        rm -rf "$dataroot/bounding_box_train_pose"
    fi
    if [ -d "$dataroot/bounding_box_train_cloth" ]; then
        rm -rf "$dataroot/bounding_box_train_cloth"
    fi
    if [ -d "$dataroot/bounding_box_train_cloth_pose" ]; then
        rm -rf "$dataroot/bounding_box_train_cloth_pose"
    fi
    if [ -d "$dataroot/bounding_box_train_pose_cloth" ]; then
        rm -rf "$dataroot/bounding_box_train_pose_cloth"
    fi
    rm "$dataroot/market-gen-guide.csv"
}

DeleteEnv () {
    root=$1
    if [ -d "$root/poseenv" ]; then
        rm -rf "$root/poseenv" 
    fi
    if [ -d "$root/clothenv" ]; then
        rm -rf "$root/clothenv"
    fi
}

DeleteGeneratedDataset $1
DeleteEnv $2