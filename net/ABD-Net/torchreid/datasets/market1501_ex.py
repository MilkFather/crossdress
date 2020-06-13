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

        market1501_data = market1501_extra.split('+')
        if verbose:
            print("Market1501_EX: Using data from", market1501_data)

        self.train_dir = []
        if 'real' in market1501_data:
            self.train_dir.append(osp.join(self.dataset_dir, 'bounding_box_train'))
        if 'pose' in market1501_data:
            self.train_dir.append(osp.join(self.dataset_dir, 'bounding_box_train_pose'))
        if 'cloth' in market1501_data:
            self.train_dir.append(osp.join(self.dataset_dir, 'bounding_box_train_cloth'))
        if 'pose-cloth' in market1501_data:
            self.train_dir.append(osp.join(self.dataset_dir, 'bounding_box_train_pose_cloth'))
        if 'cloth-pose' in market1501_data:
            self.train_dir.append(osp.join(self.dataset_dir, 'bounding_box_train_cloth_pose'))

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

        # Hardcoded information for train pids and train cams
        self.num_train_pids = 751
        self.num_train_cams = 6
        _, self.num_train_imgs, _ = self.get_imagedata_info(self.train)
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

        for _dir in dir_path:
            img_paths = glob.glob(osp.join(_dir, '*.jpg'))

            pid_container = set()
            for img_path in img_paths:
                if pattern.search(img_path) is not None:
                    pid, _ = map(int, pattern.search(img_path).groups())
                elif pattern2.search(img_path) is not None:
                    pid = -2
                if pid == -1 and os.environ.get('junk') is None:
                    continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img_path in img_paths:
                if pattern.search(img_path) is not None:
                    pid, camid = map(int, pattern.search(img_path).groups())
                elif pattern2.search(img_path) is not None:
                    pid = -2
                    camid = -1  # pseudo camera id
                if pid == -1 and os.environ.get('junk') is None:
                    continue  # junk images are just ignored
                assert -2 <= pid <= 1501  # pid == 0 means background, pid == -2 means a generated image
                if camid is int:
                    assert -1 <= camid <= 6
                    camid -= 1  # index starts from 0
                if relabel:
                    pid = pid2label[pid]
                dataset.append((img_path, pid, camid))

        return dataset