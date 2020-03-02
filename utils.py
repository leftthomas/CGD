import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044],
            'isc': [0.8324, 0.8109, 0.8041]}
rgb_std = {'car': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095],
           'isc': [0.2206, 0.2378, 0.2444]}


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[
            'train' if data_type == 'train_ext' else data_type]
        class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((252, 252)), transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
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


def recall(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    if gallery_vectors is None:
        gallery_vectors = feature_vectors.t().contiguous()
    else:
        gallery_vectors = gallery_vectors.t().contiguous()

    sim_matrix = feature_vectors.mm(gallery_vectors)

    if gallery_labels is None:
        sim_matrix[torch.eye(num_features, device=feature_vectors.device).bool()] = -1
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = sim_matrix.argsort(dim=-1, descending=True)
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    @staticmethod
    def get_anchor_positive_triplet_mask(target):
        mask = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask[torch.eye(target.size(0), device=target.device).bool()] = False
        return mask

    @staticmethod
    def get_anchor_negative_triplet_mask(target):
        labels_equal = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = ~ labels_equal
        return mask

    def forward(self, x, target):
        pairwise_dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        mask_anchor_positive = self.get_anchor_positive_triplet_mask(target).float()
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(target).float()
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        return triplet_loss.mean()
