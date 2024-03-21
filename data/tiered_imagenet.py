from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

# Set the appropriate paths of the datasets here.
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, 'E:\deeplearning\datasets\\tiered-imagenet-kwon')


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


def load_data1(file):
    file = file[:-3] + 'pth'
    re = torch.load(file)
    return re


file_path = {'train': [os.path.join(IMAGE_PATH, 'train_images.npz'), os.path.join(IMAGE_PATH, 'train_labels.pkl')],
             'val': [os.path.join(IMAGE_PATH, 'val_images.npz'), os.path.join(IMAGE_PATH, 'val_labels.pkl')],
             'test': [os.path.join(IMAGE_PATH, 'test_images.npz'), os.path.join(IMAGE_PATH, 'test_labels.pkl')]}

id2wnids = [x.strip() for x in open(os.path.join(IMAGE_PATH, 'find_labels.csv'), 'r').readlines()]
id2wnids = [w.split(',')[1] for w in id2wnids]


class tieredImageNet(data.Dataset):
    def __init__(self, setname, augment=False):
        assert (setname == 'train' or setname == 'val' or setname == 'test')
        image_path = file_path[setname][0]
        label_path = file_path[setname][1]
        data_train = load_data(label_path)
        labels = data_train['labels']
        self.data = np.load(image_path)['images']
        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        self.label = label
        self.num_class = len(set(label))
        self.label2catname = {k: id2wnids[v] for k, v in enumerate(self.wnids)}
        self.class_to_idx = {id2wnids[v]: k for k, v in enumerate(self.wnids)}
        self.targets = label

        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.ToTensor(),
            ]

        # Transformation
        self.transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(img))

        return img, label

    def __len__(self):
        return len(self.data)
