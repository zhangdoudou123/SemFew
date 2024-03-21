import math
import os
import random
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torchvision import transforms

EPS = 1e-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def proto_classifier(prototypes, query_x):
    query_x = query_x.unsqueeze(1)
    prototypes = prototypes.unsqueeze(0)
    dist = -(query_x - prototypes).pow(2).sum(dim=2)
    predict = torch.argmax(dist, dim=1)
    return dist, predict


def Cosine_classifier(support, query, temperature=1):
    """Cosine classifier"""
    # normalize for cosine distance
    # l = cosine_similarity(query.unsqueeze(1), support.unsqueeze(0), -1)
    proto = F.normalize(support, dim=-1)
    query = F.normalize(query, dim=-1)
    logits = torch.mm(query, proto.permute([1, 0])) / temperature
    predict = torch.argmax(logits, dim=1)
    return logits, predict


def LR(support, support_y, query):
    support = support.detach().cpu().numpy()
    support_y = support_y.detach().cpu().numpy()
    query = query.detach().cpu().numpy()
    clf = LogisticRegression(penalty='l2',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(support, support_y)
    predict = clf.predict(query)
    return torch.tensor(predict)


def normalize(x, epsilon=EPS):
    # x[n, d]
    x = x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)
    return x


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_95acc(accuracies):
    acc_avg = np.mean(np.array(accuracies))
    acc_ci95 = 1.96 * np.std(np.array(accuracies)) / np.sqrt(len(accuracies))
    return acc_avg, acc_ci95


def count_kacc(proto, com_proto, query, k, args, classify=Cosine_classifier):
    proto = k * proto + (1 - k) * com_proto
    label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)
    _, predict = classify(proto, query)
    return ((predict == label).sum() / len(label)).item()


def cluster(data, n_clusters=64, num=600):
    x = []
    label = []
    for k, v in data.items():
        x.extend(v)
        label.append(k)
    data = 0
    y = np.arange(len(label)).repeat(len(v))
    x = np.array(x)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x, y)
    k_center = kmeans.cluster_centers_
    k_label = kmeans.labels_
    center = {}
    for k in range(len(label)):
        labels = k_label[k * num:(k + 1) * num]
        counts = np.bincount(labels)
        index = np.argmax(counts)
        center[label[k]] = k_center[index]
    return center


transform_train = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
])

transform_val = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
])

resize = 224
transform_train_224_cifar = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

transform_val_224_cifar = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

transform_train_224 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

transform_val_224 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

Normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train_cifar = transforms.Compose([
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Normalize
])

transform_val_cifar = transforms.Compose([
    transforms.ToTensor(),
    Normalize
])


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
