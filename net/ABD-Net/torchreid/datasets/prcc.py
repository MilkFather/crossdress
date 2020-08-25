from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from matplotlib.pyplot import imsave

from .bases import BaseImageDataset

import json

import random


class PRCC(BaseImageDataset):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    """
    dataset_dir = 'prcc'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        #self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        #self.query_dir = osp.join(self.dataset_dir, 'query')
        #self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        #self._check_before_run()

        #self.campus_dir = osp.join(self.dataset_dir, 'campus')

        train = []

        f_train=open(osp.join(self.dataset_dir, 'train_prcc_color.txt'))
        line=f_train.readline()
        while line:

            label=int(line.split(" ")[1])
            img_name=line.split(" ")[0]
            cam_name=img_name[19]
            if cam_name == 'A':
                cam_id=1
            elif cam_name == 'B':
                cam_id=2
            else:
                cam_id=3

            img_path=osp.join(self.dataset_dir,img_name)
            train.append((img_path, label, cam_id))
            line=f_train.readline()


        f_gallery=open(osp.join(self.dataset_dir, 'test_prcc_color_A.txt'))

        gallery=[]

        gallery_list=[]

        for i in range(0,71):
            gallery_list.append([])


        line=f_gallery.readline()

        while line:
            label=int(line.split(" ")[1])
            img_name=line.split(" ")[0]
            cam_id=1
            img_path=osp.join(self.dataset_dir,img_name)
            gallery_list[label].append((img_path, label, cam_id))
            line=f_gallery.readline()

        for i in range(0,71):
            print(len(gallery_list[i]))
            shuffle = list(range(len(gallery_list[i])))
            random.shuffle(shuffle)
            gallery.append(gallery_list[i][shuffle[0]])




        query=[]
        f_query=open(osp.join(self.dataset_dir, 'test_prcc_color_C.txt'))

        line=f_query.readline()
        while line:

            label=int(line.split(" ")[1])
            img_name=line.split(" ")[0]
            cam_id=3
            img_path=osp.join(self.dataset_dir,img_name)
            query.append((img_path, label, cam_id))
            line=f_query.readline()

        #train = self._process_dir(self.train_dir, relabel=True)
        #query = self._process_dir(self.query_dir, relabel=False)
        #gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> SYSUDB loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 and os.environ.get('junk') is None:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 and os.environ.get('junk') is None:
                continue  # junk images are just ignored
            assert -1 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
