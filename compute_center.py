import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from data.tiered_imagenet import tieredImageNet
from model.res12 import Res12
import torch.utils.data
from utils import transform_val, transform_val_cifar, cluster
from utils import transform_val_224_cifar, transform_val_224

def main():
    if args.backbone == 'resnet':
        if 'ImageNet' in args.dataset:
            model = Res12(avg_pool=True).cuda()
            model_dict = model.state_dict()
            checkpoint = torch.load(args.model_path)['params']
            checkpoint = {k[8:]: v for k, v in checkpoint.items()}
            checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        elif args.dataset == 'CIFAR-FS':
            model = Res12(avg_pool=True, drop_block=False).cuda()
            model_dict = model.state_dict()
            checkpoint = torch.load(args.model_path)['params']
            checkpoint = {k[8:]: v for k, v in checkpoint.items()}
            checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        else:
            model = Res12(avg_pool=True, drop_block=False).cuda()
            model_dict = model.state_dict()
            checkpoint = torch.load(args.model_path)['params']
            checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

    elif args.backbone == 'swin':
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
        trainset = ImageFolder('/path/to/your/miniimagenet/train', transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    elif args.dataset == 'FC100':
        trainset = ImageFolder('/path/to/your/fc100/train', transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        trainset = ImageFolder('/path/to/your/cifar-fs/train', transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        trainset = tieredImageNet(setname='train', augment=False)

        if args.backbone == 'swin':
            trainset = ImageFolder('/path/to/your/tiredimagenet/train', transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                              pin_memory=True)
    idx_to_class = trainset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}
    for x, labels in tqdm(train_loader):
        labels = [idx_to_class[l.item()] for l in labels]
        with torch.no_grad():
            x = model(x.cuda())
        for i, l in enumerate(labels):
            if l in data:
                data[l].append(x[i].detach().cpu().numpy())
            else:
                data[l] = [x[i].detach().cpu().numpy()]
                
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
        'center': center_mean
    }, 'center_{}.pth'.format(args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--backbone', default='resnet',
                        choices=['resnet', 'swin'])
    args = parser.parse_args()
    print(vars(args))
    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)
    main()
