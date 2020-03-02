import argparse
import os
import shutil

import torch
from PIL import Image, ImageDraw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CGD')
    parser.add_argument('--query_img_name', default='data/car/uncropped/008055.jpg', type=str, help='query image name')
    parser.add_argument('--data_base', default='car_uncropped_fixed_random_layer1_resnet18_48_12_data_base.pth',
                        type=str, help='queried database')
    parser.add_argument('--retrieval_num', default=8, type=int, help='retrieval number')

    opt = parser.parse_args()

    QUERY_IMG_NAME, DATA_BASE, RETRIEVAL_NUM = opt.query_img_name, opt.data_base, opt.retrieval_num
    DATA_NAME = DATA_BASE.split('_')[0]

    data_base = torch.load('results/{}'.format(DATA_BASE))
    if QUERY_IMG_NAME not in data_base['{}_images'.format(DATA_TYPE)]:
        raise FileNotFoundError('{} not found'.format(QUERY_IMG_NAME))
    query_index = data_base['{}_images'.format(DATA_TYPE)].index(QUERY_IMG_NAME)
    query_image = Image.open(QUERY_IMG_NAME).convert('RGB').resize((256, 256), resample=Image.BILINEAR)
    query_label = data_base['{}_labels'.format(DATA_TYPE)][query_index]
    query_feature = data_base['{}_features'.format(DATA_TYPE)][query_index]

    gallery_images = data_base['{}_images'.format('train' if DATA_TYPE == 'train' else 'gallery')]
    gallery_labels = data_base['{}_labels'.format('train' if DATA_TYPE == 'train' else 'gallery')]
    gallery_features = data_base['{}_features'.format('train' if DATA_TYPE == 'train' else 'gallery')]

    query_label = torch.tensor(query_label)
    query_feature = query_feature.view(1, *query_feature.size()).permute(1, 0, 2).contiguous()
    gallery_labels = torch.tensor(gallery_labels)
    gallery_features = gallery_features.permute(1, 2, 0).contiguous()

    sim_matrix = query_feature.bmm(gallery_features).mean(dim=0).squeeze(dim=0)
    if (DATA_NAME is not 'isc') or (DATA_NAME is 'isc' and DATA_TYPE is 'train'):
        sim_matrix[query_index] = -1
    idx = sim_matrix.argsort(dim=-1, descending=True)

    result_path = 'results/{}'.format(QUERY_IMG_NAME.split('/')[-1].split('.')[0])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    query_image.save('{}/query_img.jpg'.format(result_path))
    for num, index in enumerate(idx[:RETRIEVAL_NUM]):
        retrieval_image = Image.open(gallery_images[index.item()]).convert('RGB') \
            .resize((256, 256), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(retrieval_image)
        retrieval_label = gallery_labels[index.item()]
        retrieval_status = (retrieval_label == query_label).item()
        retrieval_prob = sim_matrix[index.item()].item()
        if retrieval_status:
            draw.rectangle((0, 0, 255, 255), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 255, 255), outline='red', width=8)
        retrieval_image.save('{}/retrieval_img_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_prob))
