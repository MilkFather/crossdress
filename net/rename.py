# -*- coding: utf-8 -*-

# Maintenance script

import pandas as pd
import os.path as osp
import os

from progress.bar import IncrementalBar

# load market-gen-guide.csv
dataset_info = pd.read_csv(osp.join("../market-1501", "market-gen-guide.csv"), header=0)

bar = IncrementalBar(max=140000)

for i in range(140000):
    shape_file = dataset_info["shape"][i]
    pose_file = dataset_info["pose"][i]
    cloth_file = dataset_info["cloth"][i]
    phase1_file = dataset_info["phase1"][i]
    phase2_file = dataset_info["phase2"][i]

    shape = shape_file.split("_")[0]
    pose = pose_file.split("_")[0]
    cloth = cloth_file.split("_")[0]

    phase1_md5 = phase1_file.split("_")[-1]
    phase2_md5 = phase2_file.split("_")[-1]

    new_phase1 = shape + "_" + pose + "_" + cloth + "_" + "gen" + "_" + phase1_md5
    new_phase2 = shape + "_" + pose + "_" + cloth + "_" + "gen" + "_" + phase2_md5

    dataset_info["phase1"][i] = new_phase1
    dataset_info["phase2"][i] = new_phase2

    os.system("mv ../market-1501/bounding_box_train_cloth/"+phase1_file+" ../market-1501/bounding_box_train_cloth/"+new_phase1)
    os.system("mv ../market-1501/bounding_box_train_pose/"+phase1_file+" ../market-1501/bounding_box_train_pose/"+new_phase1)
    os.system("mv ../market-1501/bounding_box_train_cloth_pose/"+phase2_file+" ../market-1501/bounding_box_train_cloth/"+new_phase2)
    os.system("mv ../market-1501/bounding_box_train_pose_cloth/"+phase2_file+" ../market-1501/bounding_box_train_pose/"+new_phase2)

    bar.next()

dataset_info.to_csv(osp.join("../market-1501", "market-gen-guide.csv"), index=False)

bar.finish()