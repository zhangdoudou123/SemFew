import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

ROOT = 'E:\deeplearning\datasets\miniImageNet\images'
# ROOT = '/home/scdx/archer/data/images'

class my_dataset(Dataset):

    def __init__(self, csv_path='train', img_size=84, have_root=True, transform=None):
        lines1 = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lines = [l.split('\\')[-1] for l in lines1]
        new_line = []
        for l in lines:
            new_line.append(os.path.join(ROOT, l))

        data = []
        label = []
        lb = -1

        if have_root:
            new_line = lines1

        self.wnids = []
        self.label2catname = {}
        for l in new_line:
            img_path, wnid = l.split(',')

            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.label2catname[lb] = wnid
            data.append(img_path)
            label.append(lb)

        self.data = data
        self.label = label
        if transform is None:
            if 'train' in csv_path or 'source' in csv_path:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
