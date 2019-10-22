from detectron2.config import CfgNode as CN


def add_mbpl_config(cfg):
    """
    Add config for MBPL.
    """
    _C = cfg

    _C.MODEL.MULTIBRANCH = CN()

    # Out channel for each branch
    _C.MODEL.MULTIBRANCH.UP_SCALE = 2
