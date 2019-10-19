from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec
from detectron2.modeling.roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY
from torch import nn
from torch.nn import functional as F


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class MultiBranchHead(nn.Module):
    """
      A multi branch keypoint head containing a series of 3x3 convs, group BN, ReLU
      followed by a transpose convolution and bilinear interpolation for upsampling.
      """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(MultiBranchHead, self).__init__()
        # fmt: off
        conv_dims = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        out_channels = cfg.MODEL.MULTIBRANCH.OUT_CHANNELS * num_keypoints
        in_channels = input_shape.channels
        # fmt: on

        self.conv_group = Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        in_channels = out_channels
        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            layer_channels = num_keypoints * layer_channels
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1, groups=num_keypoints)
            self.add_module("group_conv_{}".format(idx), module)
            self.blocks.append(module)
            module = ConvTranspose2d(layer_channels, layer_channels, 4, stride=2, padding=1, groups=num_keypoints)
            self.add_module("group_deconv_{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels
        self.score_map = Conv2d(in_channels, num_keypoints, 3, stride=1, padding=1, groups=num_keypoints)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv_group(x)
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = self.score_map(x)
        return x
