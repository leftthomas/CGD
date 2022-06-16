import argparse
import os

import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def process_car_data(data_path, data_type):
    if not os.path.exists('{}/{}'.format(data_path, data_type)):
        os.mkdir('{}/{}'.format(data_path, data_type))
    train_images, test_images = {}, {}
    annotations = loadmat('{}/cars_annos.mat'.format(data_path))['annotations'][0]
    for img in tqdm(annotations, desc='process {} data for car dataset'.format(data_type)):
        img_name, img_label = str(img[0][0]), str(img[5][0][0])
        if data_type == 'uncropped':
            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
        else:
            x1, y1, x2, y2 = int(img[1][0][0]), int(img[2][0][0]), int(img[3][0][0]), int(img[4][0][0])
            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB').crop((x1, y1, x2, y2))
        save_name = '{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))
        img.save(save_name)
        if int(img_label) < 99:
            if img_label in train_images:
                train_images[img_label].append(save_name)
            else:
                train_images[img_label] = [save_name]
        else:
            if img_label in test_images:
                test_images[img_label].append(save_name)
            else:
                test_images[img_label] = [save_name]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, data_type))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')

    opt = parser.parse_args()
    process_car_data('{}/car'.format(opt.data_path), 'cropped')    