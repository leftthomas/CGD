import math
import random
from itertools import product

from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
val_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])


# random assign meta class for all classes
def assign_meta_id(meta_class_size, num_class, ensemble_size):
    assert math.pow(meta_class_size, ensemble_size) >= num_class, 'make sure meta_class_size^ensemble_size >= num_class'
    assert meta_class_size <= num_class, 'make sure meta_class_size <= num_class'

    multiple = num_class // meta_class_size
    remain = num_class % meta_class_size
    if remain != 0:
        multiple += 1

    max_try, i, assign_flag = 10, 0, False
    while i < max_try:
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

    return list(zip(*idxes))
