import math
import os
import random
from itertools import product

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# TODO
rgb_mean, rgb_std = [0.4853, 0.4965, 0.4295], [0.2237, 0.2193, 0.2568]
data_path = '/home/data/uisee'


# random assign meta class for all classes
def create_random_id(meta_class_size, num_class, ensemble_size):
    multiple = num_class // meta_class_size
    remain = num_class % meta_class_size
    if remain != 0:
        multiple += 1

    idxes = []
    for _ in range(ensemble_size):
        idx_all = []
        for _ in range(multiple):
            idx_base = [j for j in range(meta_class_size)]
            random.shuffle(idx_base)
            idx_all += idx_base

        idx_all = idx_all[:num_class]
        random.shuffle(idx_all)
        idxes.append(idx_all)

    # check assign conflict
    check_list = list(zip(*idxes))
    if len(check_list) == len(set(check_list)):
        print('random assigned labels have no conflicts')
    else:
        print('random assigned labels have conflicts')

    return idxes


# fixed assign meta class for all classes
def create_fixed_id(meta_class_size, num_class, ensemble_size):
    assert math.pow(meta_class_size, ensemble_size) >= num_class, 'make sure meta_class_size^ensemble_size >= num_class'
    assert meta_class_size <= num_class, 'make sure meta_class_size <= num_class'

    max_try, i, assign_flag = 10, 0, False
    while i < max_try:
        idxes = create_random_id(meta_class_size, num_class, ensemble_size)
        check_list = list(zip(*idxes))
        i += 1
        if len(check_list) != len(set(check_list)):
            print('try to random assign labels again ({}/{})'.format(i, max_try))
            assign_flag = False
        else:
            assign_flag = True
            break

    if not assign_flag:
        remained = set(check_list)
        idx_all = set(product(range(meta_class_size), repeat=ensemble_size))
        added = set(random.sample(idx_all - remained, num_class - len(remained)))
        idxes = list(zip(*(added | remained)))

    return idxes


class ImageReader(Dataset):

    def __init__(self, data_type, ensemble_size=None, meta_class_size=None, load_ids=False):
        # TODO
        data_dict = torch.load('data/{}/{}_data_dicts.pth'.format(data_name, crop_type))[
            'train' if data_type == 'train_ext' else data_type]
        class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize(rgb_mean, rgb_std)
        if data_type == 'train':
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(), normalize])
            ids_name = 'results/{}_{}_{}_ids.pth'.format(data_name, ensemble_size, meta_class_size)

            if load_ids:
                if os.path.exists(ids_name):
                    meta_ids = torch.load(ids_name)
                else:
                    raise FileNotFoundError('{} is not exist'.format(ids_name))
            else:
                meta_ids = create_fixed_id(meta_class_size, len(data_dict), ensemble_size)
                torch.save(meta_ids, ids_name)

            # balance data for each class
            max_size = 300
            self.images, self.labels = [], []
            for label, image_list in data_dict.items():
                if len(image_list) > max_size:
                    image_list = random.sample(image_list, max_size)
                self.images += image_list
                meta_label = []
                for meta_id in meta_ids:
                    meta_label.append(meta_id[class_to_idx[label]])
                meta_label = torch.tensor(meta_label)
                self.labels += [meta_label] * len(image_list)
        else:
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), normalize])
            self.images, self.labels = [], []
            for label, image_list in data_dict.items():
                self.images += image_list
                self.labels += [class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)
