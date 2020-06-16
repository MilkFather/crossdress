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
import pandas as pd
from scipy.misc import imsave

from .bases import BaseImageDataset


class Market1501_EX(BaseImageDataset):
    """
    Market1501_EX

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    # generated: 140000 (train)
    """
    dataset_dir = 'market-1501'

    def __init__(self, root='data', market1501_extra='real', verbose=True, **kwargs):
        super(Market1501_EX, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        # read info file, for reference
        dataset_info = pd.read_csv(osp.join(self.dataset_dir, "market-gen-guide.csv"), header=0)
        self.shape = dataset_info["shape"]
        self.pose = dataset_info["pose"]
        self.cloth = dataset_info["cloth"]
        self.phase1 = dataset_info["phase1"]
        self.phase2 = dataset_info["phase2"]

        # create some constants
        self.real_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.pose_dir = osp.join(self.dataset_dir, 'bounding_box_train_pose')
        self.cloth_dir = osp.join(self.dataset_dir, 'bounding_box_train_cloth')
        self.posecloth_dir = osp.join(self.dataset_dir, 'bounding_box_train_pose_cloth')
        self.clothpose_dir = osp.join(self.dataset_dir, 'bounding_box_train_cloth_pose')

        # make a convenient mapping from dataset string to data directory
        data_mapping = {'real': self.real_dir, 
                        'pose': self.pose_dir, 
                        'cloth': self.cloth_dir, 
                        'pose-cloth': self.posecloth_dir, 
                        'cloth-pose': self.clothpose_dir}

        # retrive the data we want
        market1501_data = market1501_extra.split('+')
        if verbose:
            print("Market1501_EX: Using data from", market1501_data)

        # push data directories into a list
        self.train_dir = []
        for data_itm in market1501_data:
            self.train_dir.append(data_mapping[data_itm])
        # we don't change how we handle query and gallery
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dirs(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501_EX loaded")
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
        for train_d in self.train_dir:
            if not osp.exists(train_d):
                raise RuntimeError("'{}' is not available".format(train_d))
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
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_dirs(self, dir_path, relabel=False):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pattern2 = re.compile(r'([-\d]+)_gen')
        dataset = []

        pid_container = set()
        for _dir in dir_path:
            img_paths = glob.glob(osp.join(_dir, '*.jpg'))

            for img_path in img_paths:
                if pattern.search(img_path) is not None:
                    pid, _ = map(int, pattern.search(img_path).groups())
                else:
                    pid = list(map(int, pattern2.search(img_path).groups()))[0]
                if pid == -1 and os.environ.get('junk') is None:
                    continue  # junk images are just ignored
                pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for _dir in dir_path:
            img_paths = glob.glob(osp.join(_dir, '*.jpg'))
                
            for img_path in img_paths:
                if _dir == self.real_dir: # use pattern 1
                    pid, camid = map(int, pattern.search(img_path).groups())
                else: # use pattern 2
                    pid = list(map(int, pattern2.search(img_path).groups()))[0]
                    camid = 1  # fake camera id, actually we don't care

                if pid == -1 and os.environ.get('junk') is None:
                    continue  # junk images are just ignored
                assert -1 <= pid <= 1501  # pid == 0 means background
                if camid is int:
                    assert 1 <= camid <= 6
                    camid -= 1  # index starts from 0
                if relabel:
                    pid = pid2label[pid]

                # we need to find out how generated files come from
                if _dir == self.real_dir:
                    gen_info = [pid]
                elif _dir == self.pose_dir:
                    origin_pose = self.pose[self.phase1.index(img_path.split('/')[-1])]
                    origin_pose_pid, _ = map(int, pattern.search(origin_pose).groups())
                    origin_pose_pid = pid2label[origin_pose_pid]
                    gen_info = [pid, origin_pose_pid]
                elif _dir == self.cloth_dir:
                    origin_cloth = self.cloth[self.phase1.index(img_path.split('/')[-1])]
                    origin_cloth_pid, _ = map(int, pattern.search(origin_cloth).groups())
                    origin_cloth_pid = pid2label[origin_cloth_pid]
                    gen_info = [pid, origin_cloth_pid]
                else:
                    origin_pose = self.pose[self.phase2.index(img_path.split('/')[-1])]
                    origin_pose_pid, _ = map(int, pattern.search(origin_pose).groups())
                    origin_pose_pid = pid2label[origin_pose_pid]
                    origin_cloth = self.cloth[self.phase2.index(img_path.split('/')[-1])]
                    origin_cloth_pid, _ = map(int, pattern.search(origin_cloth).groups())
                    origin_cloth_pid = pid2label[origin_cloth_pid]
                    gen_info = [pid, origin_pose_pid, origin_cloth, pid]
                
                # add to dataset
                dataset.append((img_path, pid, camid, gen_info))

        return dataset