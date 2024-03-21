import csv

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd()
data_path = 'F:\datasets\CUB_200_2011\CUB_200_2011\crop_images'
savedir = './'
dataset_list = ['base', 'val', 'novel']

# if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append(
        [join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])

label2name = {str(i):l for i, l in enumerate(folder_list)}

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i % 2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i % 4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i % 4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)

    with open('cub_{}.csv'.format(dataset), "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for n, k in enumerate(file_list):
            w = [k, label2name[str(label_list[n])]]
            print(w)
            writer.writerow(w)


