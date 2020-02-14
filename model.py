import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, ensemble_size, meta_class_size, backbone_type='resnet18', share_type='layer1',
                 with_random=False, with_fc=True):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnext50': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[backbone_type]
        module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # configs
        self.ensemble_size, self.with_random, self.with_fc = ensemble_size, with_random, with_fc

        # common features
        self.public_extractor, basic_model = [], backbone(pretrained=True)
        common_module_names = module_names[:module_names.index(share_type) + 1]
        for name, module in basic_model.named_children():
            if name in common_module_names:
                self.public_extractor.append(module)
        self.public_extractor = nn.Sequential(*self.public_extractor)
        print("# trainable public extractor parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.public_extractor.parameters()))

        # individual features
        self.learners, individual_module_names = [], module_names[module_names.index(share_type) + 1:]
        for i in range(ensemble_size):
            learner, basic_model = [], backbone(pretrained=True)
            for name, module in basic_model.named_children():
                if name in individual_module_names:
                    learner.append(module)
            self.learners.append(nn.Sequential(*learner))
        self.learners = nn.ModuleList(self.learners)
        print("# trainable individual learner parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.learners.parameters()) // ensemble_size)

        if with_fc:
            # individual classifiers
            self.classifiers = nn.ModuleList([nn.Linear(512 * expansion, meta_class_size)
                                              for _ in range(ensemble_size)])
            print("# trainable individual classifier parameters:",
                  sum(param.numel() if param.requires_grad else 0 for param in
                      self.classifiers.parameters()) // ensemble_size)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.public_extractor(x)
        out = []
        if self.with_random and self.training:
            branch_weight = torch.rand(batch_size, self.ensemble_size, device=x.device)
            branch_weight = F.softmax(branch_weight, dim=-1)
        else:
            branch_weight = torch.ones(batch_size, self.ensemble_size, device=x.device)
        for i in range(self.ensemble_size):
            individual_feature = self.learners[i](branch_weight[:, i].view(batch_size, 1, 1, 1) * common_feature)
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            if self.with_fc:
                global_feature = self.classifiers[i](global_feature)
            out.append(global_feature)
        out = torch.stack(out, dim=1)
        return out
