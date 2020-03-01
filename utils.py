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
    def __init__(self, margin=1.0, squared=False):
        super().__init__()
        self.margin = margin
        self.squared = squared

    def _pairwise_distances(self, inputs):
        dot_product = torch.matmul(inputs, torch.transpose(inputs, 0, 1))
        square_norm = torch.diag(dot_product)
        distances = torch.unsqueeze(square_norm, 0) - 2 * dot_product + torch.unsqueeze(square_norm, 1)

        distances = F.relu(distances)

        if not self.squared:
            distances = distances.clamp(min=1e-16)
            distances = torch.sqrt(distances)
        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        indices_equal = torch.eye(labels.shape[0])
        indices_not_equal = -1 * indices_equal + 1

        labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

        mask = indices_not_equal.byte().cuda() & labels_equal.byte().cuda()
        mask = mask.float().cuda()
        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
        labels_equal = labels_equal.float().cuda()

        mask = -1 * labels_equal + 1
        return mask

    def forward(self, input, targets):
        pairwise_dist = self._pairwise_distances(input)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(targets)
        mask_anchor_positive = mask_anchor_positive.float()

        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self._get_anchor_negative_triplet_mask(targets)
        mask_anchor_negative = mask_anchor_negative.float()

        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        dist = 1 * hardest_positive_dist - 1 * hardest_negative_dist + self.margin
        triplet_loss = F.relu(dist)

        return triplet_loss.float().sum()
