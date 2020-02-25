import torch
from torch import nn
from torch.nn import functional as F


class FastSCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.learning_to_down_sample = LearningToDownSample(in_channels)
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusion(scale_factor=4)
        self.classifier = Classifier(num_classes, scale_factor=8)

    def forward(self, x):
        shared = self.learning_to_down_sample(x)
        x = self.global_feature_extractor(shared)
        x = self.feature_fusion(shared, x)
        x = self.classifier(x)
        return x


class LearningToDownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=32, stride=2)
        self.dsconv1 = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            # pointwise convolution
            nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.dsconv2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, dilation=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_block = nn.Sequential(Bottleneck(64, 64, 2, 6),
                                         Bottleneck(64, 64, 1, 6),
                                         Bottleneck(64, 64, 1, 6))
        self.second_block = nn.Sequential(Bottleneck(64, 96, 2, 6),
                                          Bottleneck(96, 96, 1, 6),
                                          Bottleneck(96, 96, 1, 6))
        self.third_block = nn.Sequential(Bottleneck(96, 128, 1, 6),
                                         Bottleneck(128, 128, 1, 6),
                                         Bottleneck(128, 128, 1, 6))
        self.ppm = PPMModule(128, 128)

    def forward(self, x):
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.ppm(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.conv_high_res = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self.dwconv = ConvBlock(in_channels=128, out_channels=128, stride=1, padding=scale_factor,
                                dilation=scale_factor, groups=128)
        self.conv_low_res = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_res_input, low_res_input):
        low_res_input = F.interpolate(input=low_res_input, scale_factor=self.scale_factor, mode='bilinear',
                                      align_corners=True)
        low_res_input = self.dwconv(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)

        high_res_input = self.conv_high_res(high_res_input)
        x = torch.add(high_res_input, low_res_input)
        return self.relu(x)


class Classifier(nn.Module):
    def __init__(self, num_classes, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.dsconv1 = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            # pointwise convolution
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.dsconv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.drop_out = nn.Dropout(p=0.1)
        self.conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.drop_out(x)
        x = self.conv(x)
        x = F.interpolate(input=x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv(input)
        return self.relu(self.bn(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PPMModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super().__init__()

        inter_channels = in_channels // len(sizes)
        assert in_channels % len(sizes) == 0

        self.stages = nn.ModuleList([self._make_stage(in_channels, inter_channels, size) for size in sizes])
        self.conv = ConvBlock(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def _make_stage(self, in_channels, inter_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = ConvBlock(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear',
                                align_corners=True) for stage in self.stages] + [feats]
        bottle = self.conv(torch.cat(priors, dim=1))
        return bottle
