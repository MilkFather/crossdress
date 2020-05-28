import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch


class KeyGenDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.source_dir = opt.source_dir
        self.source_files = opt.source_files
        self.pose_dir = opt.pose_source_dir
        self.pose_files = opt.pose_source_files
        self.shape_files = opt.shape_files
       
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