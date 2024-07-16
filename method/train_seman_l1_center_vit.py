import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from method.SemAlign import SemAlign
from model.swin_transformer import swin_tiny
from data.samplers import CategoriesSampler
from logger import loggers
from utils import Cosine_classifier, count_95acc, count_kacc, transform_train_224_cifar, transform_val_224_cifar, \
    transform_val_224, transform_train_224

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--feat-size', type=int, default=768)
    parser.add_argument('--semantic-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'bert'])
    parser.add_argument('--text_type', type=str, default='gpt',
                        choices=['gpt', 'name', 'definition'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    args = parser.parse_args()
    args.model_path = '../checkpoint/Swin-Tiny-{}.pth'.format(args.dataset)

    args.work_dir = 'Vit_{}_{}_{}_{}'.format(args.dataset, args.mode, args.text_type, args.center)

    if os.path.exists(args.work_dir) is False:
        os.mkdir(args.work_dir)

    log = loggers(os.path.join(args.work_dir, 'train'))
    log.info(vars(args))

    writer = SummaryWriter(os.path.join(args.work_dir, 'train_semantic'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'MiniImageNet':
        args.val = 'F:\datasets\miniImageNet\\val'
        args.train = 'F:\datasets\miniImageNet\\base'
        train_dataset = ImageFolder(args.train, transform=transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val_224)
    elif args.dataset == 'FC100':
        args.val = 'F:\datasets\FC100\\val'
        args.train = 'F:\datasets\FC100\\train'
        train_dataset = ImageFolder(args.train, transform=transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        args.val = 'F:\datasets\CIFAR-FS\cifar100\\val'
        args.train = 'F:\datasets\CIFAR-FS\cifar100\\base'
        train_dataset = ImageFolder(args.train, transform=transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        args.val = 'E:\deeplearning\datasets\\tiered_imagenet\\val'
        args.train = 'E:\deeplearning\datasets\\tiered_imagenet\\train'
        train_dataset = ImageFolder(args.train, transform=transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    val_idx_to_class = val_dataset.class_to_idx
    val_idx_to_class = {k: v for v, k in val_idx_to_class.items()}
    idx_to_class = train_dataset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_sampler = CategoriesSampler(val_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    proto_center = torch.load('../division/center_{}_vit.pth'.format(args.dataset))[args.center]

    model = swin_tiny().to(device)
    model_dict = model.state_dict()
    checkpoint = torch.load(args.model_path)['params']
    checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    print(len(checkpoint))
    model.load_state_dict(checkpoint)
    model.eval()

    H = SemAlign(args.feat_size, args.semantic_size, h_size=4096, drop=args.drop).to(device)
    optimizer = torch.optim.Adam(H.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    if 'ImageNet' in args.dataset:
        semantic = torch.load('../semantic/imagenet_semantic_{}_{}.pth'.format(args.mode, args.text_type))['semantic_feature']
    else:
        semantic = torch.load('../semantic/cifar100_semantic_{}_{}.pth'.format(args.mode, args.text_type))[
            'semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}

    gap_acc = -1
    max_acc1 = 0.0
    for epoch in range(args.max_epoch):
        for step, [data, labels] in enumerate(tqdm(train_loader)):
            proto = torch.tensor(np.array([proto_center[idx_to_class[l.item()]] for l in labels])).cuda()
            text_feature = torch.stack([semantic[idx_to_class[l.item()]] for l in labels]).cuda()
            with torch.no_grad():
                img_feature = model(data.cuda())

            optimizer.zero_grad()
            H.zero_grad()
            H.train()
            fusion = H(text_feature, img_feature)
            recon_loss = F.l1_loss(fusion, proto)

            g_loss = recon_loss
            g_loss.backward()
            optimizer.step()

        log.info(
            "[Epoch %d/%d] [recon loss: %f] "
            % (epoch, args.max_epoch, recon_loss.item(),)
        )

        lr_scheduler.step()

        # val
        ks = np.arange(0, 101) * 0.01
        label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)
        if epoch % 10 == 0 or epoch > 0:
            P_acc = {}
            O_acc = []
            G_acc = []
            with torch.no_grad():
                for data, labels in tqdm(val_loader):
                    data = data.cuda()
                    data = model(data)
                    n_support = args.shot * args.test_way
                    support, query = data[:n_support], data[n_support:]

                    proto = support.reshape(args.shot, args.test_way, -1).mean(dim=0)
                    s = torch.stack([semantic[val_idx_to_class[l.item()]] for l in labels[:args.test_way]]).cuda()
                    gen_proto = H(s, support)

                    _, predict0 = Cosine_classifier(proto, query)
                    _, predict1 = Cosine_classifier(gen_proto, query)
                    O_acc.append(((predict0 == label).sum() / len(label)).item())
                    G_acc.append(((predict1 == label).sum() / len(label)).item())

                    for f in ks:
                        if str(f) in P_acc:
                            P_acc[str(f)].append(count_kacc(proto, gen_proto, query, f, args))
                        else:
                            P_acc[str(f)] = []
                            P_acc[str(f)].append(count_kacc(proto, gen_proto, query, f, args))

                O_acc, O_95 = count_95acc(np.array(O_acc))
                G_acc, G_95 = count_95acc(np.array(G_acc))

                max_acc = {
                    'k': 0,
                    'acc': 0,
                    'acc95': 0,
                }
                for k, v in P_acc.items():
                    P_acc[k] = count_95acc(np.array(v))
                    if P_acc[k][0] > max_acc['acc']:
                        max_acc['acc'] = P_acc[k][0]
                        max_acc['acc95'] = P_acc[k][1]
                        max_acc['k'] = k
                cur_gap = max_acc['acc'] - O_acc
                if cur_gap > gap_acc:
                    gap_acc = cur_gap
                if max_acc['acc'] > max_acc1:
                    max_acc1 = max_acc['acc']
                    torch.save({
                        'G': H,
                        'epoch': epoch,
                        'k': max_acc['k'],
                        'acc': max_acc1
                    }, os.path.join(args.work_dir, 'epoch_best.pth'.format(epoch)))
                    log.info('best epoch: %d' % epoch)
                    print('save', epoch)

            writer.add_scalars("add_scalars/acc", {'origin': O_acc,
                                                   'complete': max_acc['acc']}, epoch)
            log.info(
                'epoch: %d |origin acc: %.2f+%.2f%% |complete acc: %.2f+%.2f%% |gap: %.2f/%.2f |k: %s' % (
                    epoch, O_acc * 100, O_95 * 100, max_acc['acc'] * 100, max_acc['acc95'] * 100,
                    gap_acc * 100, cur_gap * 100, max_acc['k']))
            log.info('ACC |proto acc: %.2f+%.2f%% |gen acc: %.2f+%.2f%% |Max: %.2f' % (
                O_acc * 100, O_95 * 100, G_acc * 100, G_95 * 100, 100 * max_acc1))

        # torch.save({
        #     'G': generator,
        #     'epoch': epoch,
        #     'k': max_acc['k']
        # }, os.path.join(args.work_dir, 'epoch_{}.pth'.format(epoch)))
    writer.close()
