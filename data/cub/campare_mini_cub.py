import os
import numpy as np
import clip
import torch
from torch import nn
from nltk.corpus import wordnet as wn
import pandas as pd
import csv
import json
from transformers import BertTokenizer, BertModel

with open("split/label_name_to_map_mini_imagenet_full_size.json", 'r') as f:
    cub_mini_label = json.load(f)

cub_mini_label = list(cub_mini_label.keys())

cub_mini = {}
for s in cub_mini_label:
    s = s.split('/')
    if s[0] in cub_mini:
        cub_mini[s[0]].append(s[1])
    else:
        cub_mini[s[0]] = [s[1]]

mini_base = os.listdir('E:\deeplearning\datasets\miniImageNet\\train')
mini_val = os.listdir('E:\deeplearning\datasets\miniImageNet\\val')
mini_test = os.listdir('E:\deeplearning\datasets\miniImageNet\\test')

a = 1

if mini_base == cub_mini['train'] and mini_val == cub_mini['val'] and mini_test == cub_mini['test']:
    print(1)
