# -*- coding: utf-8 -*-

# This script calls the model and generates images.

import time
import os
from PIL import Image
import numpy as np
import torch
from util import util
from data.data_loader import CreateDataLoader
from options.make_options import MakeOptions
from models.models import create_model
import pandas as pd
from progress.bar import IncrementalBar

import pathlib


opt = MakeOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)
model = model.eval()

# read generation csv guide
guide_file = pd.read_csv(os.path.join(opt.dataroot, "market-gen-guide.csv"), header=0)
opt.shape_dir = os.path.join(opt.dataroot, "bounding_box_train")
opt.shape_files = guide_file["shape"]
if opt.make_phase == 1:
    opt.source_dir = os.path.join(opt.dataroot, "bounding_box_train")
    opt.source_files = guide_file["shape"]
    opt.pose_source_dir = os.path.join(opt.dataroot, "bounding_box_train")
    opt.pose_source_files = guide_file["pose"]
    opt.output_dir = os.path.join(opt.dataroot, "bounding_box_train_pose")
    opt.output_files = guide_file["phase1"]
elif opt.make_phase == 2:
    opt.source_dir = os.path.join(opt.dataroot, "bounding_box_train_cloth")
    opt.source_files = guide_file["phase1"]
    opt.pose_source_dir = os.path.join(opt.dataroot, "bounding_box_train")
    opt.pose_source_files = guide_file["pose"]
    opt.output_dir = os.path.join(opt.dataroot, "bounding_box_train_cloth_pose")
    opt.output_files = guide_file["phase2"]
else:
    raise Exception("Unknown generation phase for Pose-Transfer")

# dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# test
print("Making dataset phase", opt.make_phase)
bar = IncrementalBar(max=len(dataset), message="%(index)d/%(max)d", suffix="%(elapsed_td)s/%(eta_td)s")

pathlib.Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

for i, data in enumerate(dataset):
    if i >= len(dataset):
        break
    model.set_input(data)
    model.test()
    PP_score = model.get_D_PP()
    PB_score = model.get_D_PB()
    im = util.tensor2im(model.fake_p2.data)
    save_path = os.path.join(opt.output_dir, opt.output_files[i])
    util.save_image(im, save_path)

    print(PP_score, PB_score)

    bar.next()
bar.finish()
