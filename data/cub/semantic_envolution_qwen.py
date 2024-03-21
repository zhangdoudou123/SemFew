# -*- coding: utf-8 -*-
import json
import csv
import logging
import os
import time
from http import HTTPStatus

import dashscope
import openai
import csv

from dashscope import Generation
from nltk.corpus import wordnet as wn


def loggers(name='log'):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    # 第二步，创建一个handler，用于写入日志文件
    logfile = name + '.txt'
    fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


imagenet_template = 'Provide a brief description of the appearance of the {} in three sentences or fewer.'

data = {}
lines = [x.strip() for x in open('cub_novel.csv', 'r').readlines()]
name = list(set([l.split(',')[1] for l in lines]))
for n in name:
    data[n] = imagenet_template.format(n.split('.')[1].replace('_', ' '))

dashscope.api_key = 'sk-707d170d5ce34f8487dd60f556ef2b3a'

log = loggers('cub_prompt_qwen')

se = {}

for i, (k, v) in enumerate(data.items()):
    log.info(i)
    is_continue = True
    while is_continue:
        try:
            response = Generation.call(
                model='qwen-turbo',
                prompt=v
            )
            if response.status_code == HTTPStatus.OK:
                se[k] = response.output
                print(response.output)  # The output text
                print(response.usage)  # The usage information
                is_continue=False
            else:
                print(response.code)  # The error code.
                print(response.message)  # The error message.
        except:
            is_continue = True


with open('cub_qwen.csv', "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for n, (k, v) in enumerate(se.items()):
        w = [k, v, data[k]]
        print(w)
        writer.writerow(w)
