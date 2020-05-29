from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

from PIL import Image

import os
import os.path
import sys


class ImageListDataset(VisionDataset):
    
    def __init__(self, root, filels, loader=default_loader, transform=None, target_transform=None, is_valid_file=None):
        super(ImageListDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        samples = [(filels[i], filels[i].split("_")[0]) for i in range(len(filels))]
        
        self.root = root
        self.loader = loader
        self.samples = samples
        self.imgs = self.samples


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
