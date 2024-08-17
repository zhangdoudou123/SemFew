import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from torchvision.datasets import ImageFolder

# from data.tiered_imagenet import tieredImageNet
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.samplers import CategoriesSampler
from logger import loggers
from model.res12 import Res12
from model.swin_transformer import swin_tiny
from utils import set_seed, Cosine_classifier, count_95acc, count_kacc, transform_val_cifar, transform_val, transform_val_224, transform_val_224_cifar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'mean'])
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--feat-size', type=int, default=640)
    parser.add_argument('--semantic-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'bert'])
    parser.add_argument('--text_type', type=str, default='gpt',
                        choices=['gpt', 'name', 'definition'])
    parser.add_argument('--dataset', type=str, default='TieredImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'swin'])
    args = parser.parse_args()
    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)
    args.work_dir = '{}_{}_{}_{}_{}_{}'.format(args.backbone, args.dataset, args.mode, args.text_type, args.center, args.shot)

    log = loggers('test_{}'.format(args.dataset))
    log.info(vars(args))
    set_seed(args.seed)

    if args.dataset == 'TieredImageNet':
        args.num_workers = 0

    if args.dataset == 'MiniImageNet':
        args.test = '/path/to/your/miniimagent/test'
        test_dataset = ImageFolder(args.test, transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    elif args.dataset == 'FC100':
        args.test = '/path/to/your/fc100/test'
        test_dataset = ImageFolder(args.test, transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        args.test = '/path/to/your/cifar-fs/test'
        test_dataset = ImageFolder(args.test, transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        test_dataset = tieredImageNet(setname='test')

        if args.backbone == 'resnet':
            args.test = '/path/to/your/tiredimagenet/test'
            test_dataset = ImageFolder(args.test, transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    idx_to_class = test_dataset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}

    val_sampler = CategoriesSampler(test_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=test_dataset, batch_sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path)['params']
        checkpoint = {k[8:]: v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        
    print(len(checkpoint))
    model.load_state_dict(checkpoint)
    model.eval()

    Model_PATH = os.path.join(args.work_dir, 'epoch_best.pth')
    H = torch.load(Model_PATH)
    fusion = H['G']
    best_epoch = H['epoch']
    best_acc = H['acc']
    best_k = H['k']
    log.info('best epoch: %d %2f' % (best_epoch, best_acc * 100))
    log.info('best k: %2f' % (float(best_k)))

    if 'ImageNet' in args.dataset:
        semantic = torch.load('./semantic/imagenet_semantic_{}_{}.pth'.format(args.mode, args.text_type))['semantic_feature']
    else:
        semantic = torch.load('./semantic/cifar100_semantic_{}_{}.pth'.format(args.mode, args.text_type))['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}

    ks = np.arange(0, 101) * 0.01
    label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)
    with torch.no_grad():
        A_acc = []
        P_acc = []
        G_acc = []
        for data, labels in tqdm(val_loader):
            data = data.to(device)
            data = model(data).view(data.size(0), -1)
            n_support = args.shot * args.test_way
            support, query = data[:n_support], data[n_support:]

            proto = support.reshape(args.shot, args.test_way, -1).mean(dim=0)
            s = torch.stack([semantic[idx_to_class[l.item()]] for l in labels[:n_support]]).to(device)
            gen_proto = fusion(s, support)
            gen_proto = gen_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            dist0, predict0 = Cosine_classifier(proto, query)
            dist1, predict1 = Cosine_classifier(gen_proto, query)
            P_acc.append(((predict0 == label).sum() / len(label)).item())
            G_acc.append(((predict1 == label).sum() / len(label)).item())

            A_acc.append(count_kacc(proto, gen_proto, query, torch.tensor(float(best_k)), args))

        P_acc, P_95 = count_95acc(np.array(P_acc))
        G_acc, G_95 = count_95acc(np.array(G_acc))
        A_acc = count_95acc(np.array(A_acc))
        
        log.info('max |k: %16s |mix acc: %.2f+%.2f%% |gap: %.2f' % (
            best_k, A_acc[0] * 100, A_acc[1] * 100, A_acc[0] * 100 - P_acc * 100))
        log.info('ACC:|proto acc: %.2f+%.2f%% |gen acc: %.2f+%.2f%%' % (
            P_acc * 100, P_95 * 100, G_acc * 100, G_95 * 100))
