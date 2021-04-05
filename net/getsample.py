import os
import random
import pandas as pd
from PIL import Image

dataroot = '../data/market-1501'
orig = os.path.join(dataroot, 'bounding_box_train')
orig_pose = os.path.join(dataroot, 'bounding_box_train_pose')
orig_cloth = os.path.join(dataroot, 'bounding_box_train_cloth')
orig_pose_cloth = os.path.join(dataroot, 'bounding_box_train_pose_cloth')
orig_cloth_pose = os.path.join(dataroot, 'bounding_box_train_cloth_pose')

guide_file_path = os.path.join(dataroot, 'market-gen-guide.csv')

# read guide
guide_file = pd.read_csv(guide_file_path, header=0)

shapes = guide_file['shape']
poses = guide_file['pose']
cloths = guide_file['cloth']
phase1s = guide_file['phase1']
phase2s = guide_file['phase2']

# get files
idx = random.randint(0, len(shapes) - 1)

orig_filepath = os.path.join(orig, shapes[idx])
pose_filepath = os.path.join(orig, poses[idx])
cloth_filepath = os.path.join(orig, cloths[idx])
orig_pose_filepath = os.path.join(orig_pose, phase1s[idx])
orig_cloth_filepath = os.path.join(orig_cloth, phase1s[idx])
orig_pose_cloth_filepath = os.path.join(orig_pose_cloth, phase2s[idx])
orig_cloth_pose_filepath = os.path.join(orig_cloth_pose, phase2s[idx])

orig_file = Image.open(orig_filepath)
pose_file = Image.open(pose_filepath)
cloth_file = Image.open(cloth_filepath)
orig_pose_file = Image.open(orig_pose_filepath)
orig_cloth_file = Image.open(orig_cloth_filepath)
orig_pose_cloth_file = Image.open(orig_pose_cloth_filepath)
orig_cloth_pose_file = Image.open(orig_cloth_pose_filepath)

# assemble images
assemble = Image.new('RGB', (7 * 64, 128))
assemble.paste(orig_file, (0 * 64, 0))
assemble.paste(pose_file, (1 * 64, 0))
assemble.paste(cloth_file, (2 * 64, 0))
assemble.paste(orig_pose_file, (3 * 64, 0))
assemble.paste(orig_cloth_file, (4 * 64, 0))
assemble.paste(orig_pose_cloth_file, (5 * 64, 0))
assemble.paste(orig_cloth_pose_file, (6 * 64, 0))

# save image
assemble.save('saved_{0}.png'.format(idx))