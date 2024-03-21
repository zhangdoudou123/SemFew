import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.utils.data

from model.swin_transformer import swin_tiny
from utils import cluster, transform_val_224_cifar, transform_val_224


def main():
    model = swin_tiny().cuda()
    model_dict = model.state_dict()
    checkpoint = torch.load(args.model_path)['params']
    checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    print(len(checkpoint))
    model.load_state_dict(checkpoint)
    model.eval()

    data = {}
    batch_size = 128
    shuffle = True
    # train
    if args.dataset == 'MiniImageNet':
        trainset = ImageFolder('E:\deeplearning\datasets\miniImageNet\\train', transform=transform_val_224)
    elif args.dataset == 'FC100':
        trainset = ImageFolder('F:\datasets\FC100\\train', transform=transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        trainset = ImageFolder('F:\datasets\CIFAR-FS\cifar100\\base', transform=transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        trainset = ImageFolder('E:\deeplearning\datasets\\tiered_imagenet\\train', transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8,
                              pin_memory=True)

    idx_to_class = trainset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}
    for x, labels, index in tqdm(train_loader):
        labels = [idx_to_class[l.item()] for l in labels]
        with torch.no_grad():
            x = model(x.cuda())
        for i, l in enumerate(labels):
            if l in data:
                data[l].append(x[i].detach().cpu().numpy())
            else:
                data[l] = []
                data[l].append(x[i].detach().cpu().numpy())
    print('Finished train')

    center_mean = {}
    for k, v in data.items():
        center_mean[k] = np.array(v).mean(0)

    if args.dataset == 'TieredImageNet':
        data = {k: v[:700] for k, v in data.items()}
        center_cluster = cluster(data, len(data), 700)
    else:
        center_cluster = cluster(data, len(data), 600)

    torch.save({
        'mean': center_mean,
        'cluster': center_cluster,
    }, 'center_{}_vit.pth'.format(args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FC100',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    args = parser.parse_args()
    print(vars(args))
    args.model_path = '../checkpoint/Swin-Tiny-{}.pth'.format(args.dataset)
    main()
