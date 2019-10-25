from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, interpolate
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
        self.up_scale = cfg.MODEL.MULTIBRANCH.UP_SCALE
        conv_dims = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        in_channels = input_shape.channels
        # fmt: on

        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn_1_{}".format(idx), module)
            self.blocks.append(module)
            module = Conv2d(layer_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn_2_{}".format(idx), module)
            self.blocks.append(module)
            module = Conv2d(layer_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn_3_{}".format(idx), module)
            self.blocks.append(module)
            if idx < len(conv_dims):
                module = ConvTranspose2d(layer_channels, layer_channels, 4, stride=2, padding=1)
                self.add_module("conv_dcn_{}".format(idx), module)
                self.blocks.append(module)
                in_channels = layer_channels
            else:
                module = ConvTranspose2d(layer_channels, num_keypoints, 3, stride=1, padding=1)
                self.add_module("conv_dcn_{}".format(idx), module)
                self.blocks.append(module)
                in_channels = num_keypoints

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for idx, layer in enumerate(self.blocks, 1):
            if idx < len(self.blocks):
                x = F.relu(layer(x))
            else:
                x = layer(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x
