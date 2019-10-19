from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY
from torch import nn


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class MultiBranchHead(nn.Module):
    """
      A multi branch keypoint head containing a series of 3x3 convs, group BN, ReLU
      followed by a transpose convolution and bilinear interpolation for upsampling.
      """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(MultiBranchHead, self).__init__()
        # fmt: off
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        # fmt: on
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        pass
