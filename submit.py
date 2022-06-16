
import argparse
import os
import sys
import yaml
from collections import OrderedDict

import numpy as np
import pandas as pd
import torchvision.models as models
from tqdm import tqdm
import torch
from sklearn.preprocessing import normalize

from data.dataset import CarsDataset
from data.augmentations import get_val_aug
from utils import convert_dict_to_tuple
from model import Model


def main(args: argparse.Namespace) -> None:
    backbone_type = args.backbone_type
    gd_config = args.gd_config
    feature_dim = args.feature_dim
    
    with open(args.exp_cfg) as f:
        data = yaml.safe_load(f)
    exp_cfg = convert_dict_to_tuple(data)

    with open(args.inference_cfg) as f:
        data = yaml.safe_load(f)
    inference_cfg = convert_dict_to_tuple(data)

    # getting model and checkpoint
    print('Creating model and loading checkpoint')
    model = Model(backbone_type, gd_config, feature_dim, num_classes=1716)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    model.cuda()
    print('Weights are loaded')

    test_dataset = CarsDataset(root=inference_cfg.root,
                               annotation_file=inference_cfg.test_list,
                               transforms=get_val_aug(exp_cfg),
                               is_inference=True)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=inference_cfg.batch_size,
            shuffle=False,
            pin_memory=True)

    print('Calculating embeddings')
    with torch.no_grad():
        for i, images in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
            images = images.to('cuda')
            features, classes = model(images)
            features = features.data.cpu().numpy()

            if i == 0:
                embeddings = features
            else:
                embeddings = np.vstack((embeddings, features))

    print(embeddings.shape)

    imgname2idx_dict = dict([x[0], i] for i, x in enumerate(test_dataset.imlist))

    #normalize and get distances
    print('Normalizing and calculating distances')
    embeddings = normalize(embeddings)

    submit_pairs_df = pd.read_csv(inference_cfg.pairs_list)
    dist_arr = np.empty(len(submit_pairs_df), dtype=float)
    for row in submit_pairs_df.itertuples():
        embedding1 = embeddings[imgname2idx_dict[row.img1]]
        embedding2 = embeddings[imgname2idx_dict[row.img2]]
        dist_arr[row.Index] = np.linalg.norm(embedding1 - embedding2)

    submit_pairs_df['dist'] = dist_arr

    print('Getting final scores')
    submit_pairs_df['score'] = submit_pairs_df.dist.apply(lambda x: (2 - x)/2)

    exp_name = exp_cfg.exp_name
    model_epoch = os.path.basename(args.checkpoint_path).replace('.pth', '')
    save_path = f'./{exp_name}_{model_epoch}.csv'
    submit_pairs_df[['id', 'score']].to_csv(save_path, index=False)

    print(f'Congrats! You have created submission. {save_path}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_cfg', type=str, default='config/baseline_mcs.yml',
                        help='Path to experiment config file.')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        help='Path to checkpoint file.')
    parser.add_argument('--inference_cfg',
                        type=str,
                        default='config/inference_config.yml',
                        help='Path to inference config file.')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnext50'],
                        help='backbone network type')
    parser.add_argument('--gd_config', default='SG', type=str,
                        choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'],
                        help='global descriptors config')
    parser.add_argument('--feature_dim', default=1536, type=int, help='feature dim')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))