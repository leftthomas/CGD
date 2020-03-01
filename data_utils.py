import argparse
import os

import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def read_txt(path, data_num):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        if data_num == 2:
            data_1, data_2 = line.split()
        else:
            data_1, data_2, data_3, data_4, data_5 = line.split()
            data_2 = [data_2, data_3, data_4, data_5]
        data[data_1] = data_2
    return data


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


def process_cub_data(data_path, data_type):
    if not os.path.exists('{}/{}'.format(data_path, data_type)):
        os.mkdir('{}/{}'.format(data_path, data_type))
    images = read_txt('{}/images.txt'.format(data_path), 2)
    labels = read_txt('{}/image_class_labels.txt'.format(data_path), 2)
    bounding_boxes = read_txt('{}/bounding_boxes.txt'.format(data_path), 5)
    train_images, test_images = {}, {}
    for img_id, img_name in tqdm(images.items(), desc='process {} data for cub dataset'.format(data_type)):
        if data_type == 'uncropped':
            img = Image.open('{}/images/{}'.format(data_path, img_name)).convert('RGB')
        else:
            x1, y1 = int(float(bounding_boxes[img_id][0])), int(float(bounding_boxes[img_id][1]))
            x2, y2 = x1 + int(float(bounding_boxes[img_id][2])), y1 + int(float(bounding_boxes[img_id][3]))
            img = Image.open('{}/images/{}'.format(data_path, img_name)).convert('RGB').crop((x1, y1, x2, y2))
        save_name = '{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))
        img.save(save_name)
        if int(labels[img_id]) < 101:
            if labels[img_id] in train_images:
                train_images[labels[img_id]].append(save_name)
            else:
                train_images[labels[img_id]] = [save_name]
        else:
            if labels[img_id] in test_images:
                test_images[labels[img_id]].append(save_name)
            else:
                test_images[labels[img_id]] = [save_name]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, data_type))


def process_sop_data(data_path):
    if not os.path.exists('{}/uncropped'.format(data_path)):
        os.mkdir('{}/uncropped'.format(data_path))
    train_images, test_images = {}, {}
    data_tuple = {'train': train_images, 'test': test_images}
    for data_type, image_list in data_tuple.items():
        for index, line in enumerate(open('{}/Ebay_{}.txt'.format(data_path, data_type), 'r', encoding='utf-8')):
            if index != 0:
                _, label, _, img_name = line.split()
                img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
                save_name = '{}/uncropped/{}'.format(data_path, os.path.basename(img_name))
                img.save(save_name)
                if label in image_list:
                    image_list[label].append(save_name)
                else:
                    image_list[label] = [save_name]
    torch.save({'train': train_images, 'test': test_images}, '{}/uncropped_data_dicts.pth'.format(data_path))


def process_isc_data(data_path):
    if not os.path.exists('{}/uncropped'.format(data_path)):
        os.mkdir('{}/uncropped'.format(data_path))
    train_images, query_images, gallery_images = {}, {}, {}
    for index, line in enumerate(open('{}/Eval/list_eval_partition.txt'.format(data_path), 'r', encoding='utf-8')):
        if index > 1:
            img_name, label, status = line.split()
            img = Image.open('{}/Img/{}'.format(data_path, img_name)).convert('RGB')
            save_name = '{}/uncropped/{}_{}'.format(data_path, img_name.split('/')[-2], os.path.basename(img_name))
            img.save(save_name)
            if status == 'train':
                if label in train_images:
                    train_images[label].append(save_name)
                else:
                    train_images[label] = [save_name]
            elif status == 'query':
                if label in query_images:
                    query_images[label].append(save_name)
                else:
                    query_images[label] = [save_name]
            elif status == 'gallery':
                if label in gallery_images:
                    gallery_images[label].append(save_name)
                else:
                    gallery_images[label] = [save_name]

    torch.save({'train': train_images, 'query': query_images, 'gallery': gallery_images},
               '{}/uncropped_data_dicts.pth'.format(data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')

    opt = parser.parse_args()

    process_car_data('{}/car'.format(opt.data_path), 'uncropped')
    process_car_data('{}/car'.format(opt.data_path), 'cropped')
    process_cub_data('{}/cub'.format(opt.data_path), 'uncropped')
    process_cub_data('{}/cub'.format(opt.data_path), 'cropped')
    print('processing sop dataset')
    process_sop_data('{}/sop'.format(opt.data_path))
    print('processing isc dataset')
    process_isc_data('{}/isc'.format(opt.data_path))
