import os

import pandas as pd
import cv2
from PIL import Image
import torch.utils.data as data


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class CarsDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file).values.tolist()
        self.transforms = transforms
        self.is_inference = is_inference

    def __getitem__(self, index):
        cv2.setNumThreads(6)
        if self.is_inference:
            impath, x1, y1, x2, y2 = self.imlist[index]
        else:
            impath, x1, y1, x2, y2, target = self.imlist[index]
        full_imname = os.path.join(self.root, impath)

        if not os.path.exists(full_imname):
            print('No file ', full_imname)

        img = read_image(full_imname)

        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))

        if 0 <= x1 < x2 and 0 <= y1 < y2 and 0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]:
            img = img[y1: y2, x1: x2]

        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)
