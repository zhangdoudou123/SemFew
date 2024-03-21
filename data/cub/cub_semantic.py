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

text_type = 'gpt3'
imagenet_template = 'This is a photo of a bird called {}.'

data = {}
if text_type == 'qwen':
    with open('cub_qwen.csv', "r", newline='', ) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            k = str(int(line[0][:3]))
            data[k] = json.loads(line[1])['text']
elif text_type == 'name':
    with open('F:\datasets\CUB_200_2011\CUB_200_2011\classes.txt', "r", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line = line[0].split(' ')
            data[line[0]] = imagenet_template.format(line[1].split('.')[1].replace('_', ' '))

    data = {k: v.replace('_', '') for k, v in data.items()}
elif text_type == 'gpt3':
    with open('cub_gpt3.csv', "r", newline='', ) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            k = str(int(line[0][:3]))
            data[k] = line[1]


# clip
print(clip.available_models())
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16',512 - 768 'ViT-L/14', 'ViT-L/14@336px']

model, preprocess = clip.load("ViT-B/16")

attr_dict = {}
with torch.no_grad():
    zeroshot_weights = []
    for i, (k, v) in enumerate(data.items()):
        print(i)
        print(k)
        # print(v)
        texts = v.split('.')
        texts = [t + '.' for t in texts if len(t) > 0]
        texts = ' '.join(texts)
        print(texts)
        texts = clip.tokenize(texts, truncate=True)  # tokenize
        class_embeddings = model.encode_text(texts.cuda())  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        attr_dict[k] = class_embedding.detach().cpu()
        # attr_dict[k] = class_embeddings.squeeze().detach().cpu()

torch.save({'semantic_feature': attr_dict}, 'cub_semantic_clip_{}.pth'.format(text_type))

print('clip')

# bert
tokenizer = BertTokenizer.from_pretrained('E:/deeplearning/model/BERT/')
model = BertModel.from_pretrained("E:/deeplearning/model/BERT")

attr_dict = {}
with torch.no_grad():
    for i, (k, v) in enumerate(data.items()):
        print(i)
        print(v)
        encoded_input = tokenizer(v, return_tensors='pt')
        output = model(**encoded_input).pooler_output.squeeze()
        attr_dict[k] = output.detach().cpu()
torch.save({'semantic_feature': attr_dict}, 'cub_semantic_bert_{}.pth'.format(text_type))
