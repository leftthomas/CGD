import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # bias and running_mean are the key values to change the results!
                nn.init.normal_(m.bias)
                nn.init.normal_(m.running_mean)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, ensemble_size, with_random=False):
        super(Model, self).__init__()

        # configs
        self.ensemble_size, self.with_random = ensemble_size, with_random

        # common features
        self.public_extractor = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(8), nn.ReLU(inplace=True))

        # individual features
        self.learners = []
        for i in range(ensemble_size):
            self.learners.append(nn.Sequential(ConvBlock(8, 8)))
        self.learners = nn.ModuleList(self.learners)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.public_extractor(x)
        out = []
        if self.with_random:
            branch_weight = torch.rand(batch_size, self.ensemble_size, device=x.device)
            branch_weight = F.softmax(branch_weight, dim=-1)
        else:
            branch_weight = torch.ones(batch_size, self.ensemble_size, device=x.device)
        for i in range(self.ensemble_size):
            individual_feature = self.learners[i](branch_weight[:, i].view(batch_size, 1, 1, 1) * common_feature)
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            out.append(global_feature)
        out = torch.stack(out, dim=1)
        return out


data = torch.randn(10, 3, 5, 5)

model = Model(ensemble_size=2, with_random=True)
model.eval()

batch_sizes = [2, 4, 8]
sim_matrix = []

for batch_size in batch_sizes:
    output = []
    for item in torch.split(data, split_size_or_sections=batch_size):
        out = model(item)
        output.append(F.normalize(out, dim=-1))
    output = torch.cat(output, dim=0)
    sim_matrix.append(torch.bmm(output.permute(1, 0, 2).contiguous(), output.permute(1, 2, 0).contiguous()))

for m1 in sim_matrix:
    for m2 in sim_matrix:
        if not torch.allclose(m1, m2):
            print('fail!')
