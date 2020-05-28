# -*- coding: utf-8 -*-

# This script calls the model and generates images.

import time
import os
from PIL import Image
import numpy as np
import torch
from util import util
from data.base_dataset import BaseDataset, get_transform
from options.make_options import MakeOptions
from models.models import create_model
import pandas as pd
from progress.bar import IncrementalBar

class MakeDataset(BaseDataset):
    def initialize(self, opt, source_dir, source_files, pose_dir, pose_files, shape_dir, shape_files):
        self.opt = opt

        self.source_dir = source_dir
        self.source_files = source_files
        self.pose_dir = pose_dir
        self.pose_files = pose_files
        self.shape_files = shape_files
       
        self.dir_K = os.path.join(opt.dataroot, 'bounding_box_trainK') #keypoints

        self.transform = get_transform(opt)
    
    def __getitem__(self, index):
        P1_name = self.source_files[index]
        P2_name = self.pose_files[index]
        
        P1_path = os.path.join(self.source_dir, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, self.shape_files[index] + '.npy') # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.pose_dir, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path) 
        
        BP1 = torch.from_numpy(BP1_img).float() #h, w, c
        BP1 = BP1.transpose(2, 0) #c,w,h
        BP1 = BP1.transpose(2, 1) #c,h,w 

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0) #c,w,h
        BP2 = BP2.transpose(2, 1) #c,h,w 

        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name}

    def __len__(self):
        return len(self.source_files)

    def name(self):
        return 'GenDataset'


opt = MakeOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)
model = model.eval()

# read generation csv guide
guide_file = pd.read_csv(os.path.join(opt.dataroot, "market-gen-guide.csv"), header=0)
shape_dir = os.path.join(opt.dataroot, "bounding_box_train")
shape_files = guide_file["shape"]
if opt.make_phase == 1:
    source_dir = os.path.join(opt.dataroot, "bounding_box_train")
    source_files = guide_file["shape"]
    pose_source_dir = os.path.join(opt.dataroot, "bounding_box_train")
    pose_source_files = guide_file["pose"]
    output_dir = os.path.join(opt.dataroot, "bounding_box_train_pose")
    output_files = guide_file["phase1"]
elif opt.make_phase == 2:
    source_dir = os.path.join(opt.dataroot, "bounding_box_train_cloth")
    source_files = guide_file["phase1"]
    pose_source_dir = os.path.join(opt.dataroot, "bounding_box_train")
    pose_source_files = guide_file["pose"]
    output_dir = os.path.join(opt.dataroot, "bounding_box_train_cloth_pose")
    output_files = guide_file["phase2"]
else:
    raise Exception("Unknown generation phase for Pose-Transfer")

# dataset
dataset = MakeDataset()
dataset.initialize(opt, source_dir, source_files, pose_source_dir, pose_source_files, shape_dir, shape_files)

# test
print("Making dataset phase", opt.make_phase)
bar = IncrementalBar(max=len(dataset), message="%(index)d/%(max)d", suffix="%(elapsed_td)s/%(eta_td)s")

for i, data in enumerate(dataset):
    if i >= len(dataset):
        break
    model.set_input(data)
    model.test()
    im = util.tensor2im(model.fake_p2.data)
    save_path = os.path.join(output_dir, output_files[i])
    util.save_image(im, save_path)

    bar.next()
bar.finish()
