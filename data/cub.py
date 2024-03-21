import json
import os
import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH3 = osp.abspath(osp.join(THIS_PATH, '..'))
IMAGE_PATH = osp.join(ROOT_PATH3, 'data/cub')
SPLIT_PATH = osp.join(ROOT_PATH3, 'data/cub/split')
IMAGE_PATH = 'F:\datasets\CUB_200_2011\CUB_200_2011\crop_images'
IMAGE_PATH = 'F:\datasets\CUB_200_2011\CUB_200_2011\images'
# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)

class CUB(Dataset):

    def __init__(self, setname, augment=False):
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        self.data, self.label = self.parse_csv(txt_path)
        
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84
        
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if False:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        else:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        self.label2catname = {}
        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_PATH, wnid, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.label2catname[lb] = wnid
                
            data.append(path)
            label.append(lb)

        return data, label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))
        return image, label


class CUB2(Dataset):

    def __init__(self, csv_path='novel.csv', image_size=84):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]
        data = []
        label = []
        lb = -1

        self.wnids = []
        self.label2catname = {}
        for l in lines:
            img_path, wnid = l.split(',')

            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.label2catname[lb] = wnid
            data.append(img_path)
            label.append(lb)

        self.data = data
        self.label = label

        if 'base' in csv_path:
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if False:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        else:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

