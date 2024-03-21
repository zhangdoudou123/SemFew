import os.path

import cv2
from PIL import Image

data_path = "F:\datasets\CUB_200_2011"

imgs = os.path.join(data_path, "CUB_200_2011\images")

imgs_path = {}
with open(os.path.join(data_path, "CUB_200_2011", "images.txt"), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.split(' ')
        imgs_path[line[0]] = line[1]

bounding_box = {}
with open(os.path.join(data_path, "CUB_200_2011", "bounding_boxes.txt"), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.split(' ')
        bounding_box[line[0]] = line[1:]

for k, v in imgs_path.items():
    box = [float(d) for d in bounding_box[k]]  # <x>, <y>, <width>, and <height>
    box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    path = os.path.join(imgs, v)
    img = Image.open(path)
    print(img.size)
    cropped = img.crop(box)  # (left, upper, right, lower)
    save_path = os.path.join(data_path, "CUB_200_2011\crop_images", v)
    save_root = '\\'.join(save_path.replace('/', '\\').split('\\')[:-1])
    if os.path.exists(save_root):
        cropped.save(save_path)
    else:
        os.mkdir(save_root)
        cropped.save(save_path)
