import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class FeatureExtractor(nn.Module):
    def __init__(self, meta_class_size, ensemble_size):
        super(FeatureExtractor, self).__init__()

        # backbone
        backbone, expansion = resnet18, 1
        module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # configs
        self.ensemble_size = ensemble_size

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

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.public_extractor(x)
        out_features = []
        for i in range(self.ensemble_size):
            individual_feature = self.learners[i](common_feature)
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            classes = self.classifiers[i](global_feature)
            out_features.append(classes)
        out_features = torch.stack(out_features, dim=1)
        return out_features
