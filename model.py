import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, meta_class_size, ensemble_size, num_classes, with_random):
        super(Model, self).__init__()

        # backbone
        backbone, expansion = resnet18, 1
        module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # configs
        self.ensemble_size, self.with_random = ensemble_size, with_random

        # common features
        self.public_extractor, basic_model = [], backbone(pretrained=True)
        common_module_names = module_names[:module_names.index('layer1') + 1]
        for name, module in basic_model.named_children():
            if name in common_module_names:
                self.public_extractor.append(module)
        self.public_extractor = nn.Sequential(*self.public_extractor)

        # individual features
        self.learners, individual_module_names = [], module_names[module_names.index('layer1') + 1:]
        for i in range(ensemble_size):
            learner, basic_model = [], backbone(pretrained=True)
            for name, module in basic_model.named_children():
                if name in individual_module_names:
                    learner.append(module)
            self.learners.append(nn.Sequential(*learner))
        self.learners = nn.ModuleList(self.learners)

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Linear(512 * expansion, meta_class_size) for _ in range(ensemble_size)])
        self.classifier = nn.Linear(ensemble_size * meta_class_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.public_extractor(x)
        if self.with_random:
            branch_weight = torch.rand(self.ensemble_size, device=x.device)
            branch_weight = F.softmax(branch_weight, dim=-1)
        else:
            branch_weight = torch.ones(self.ensemble_size, device=x.device)
        out_features = []
        for i in range(self.ensemble_size):
            individual_feature = self.learners[i](branch_weight[i] * common_feature)
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            classes = self.classifiers[i](global_feature)
            out_features.append(classes)
        out_features = torch.stack(out_features, dim=1)
        out_class = self.classifier(out_features.detach().view(out_features.size(0), -1))
        return out_features, out_class
