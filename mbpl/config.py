from detectron2.config import CfgNode as CN


def add_mbpl_config(cfg):
    """
    Add config for MBPL.
    """
    _C = cfg

    _C.MODEL.TRIDENT = CN()

    # Number of branches for TridentNet.
    _C.MODEL.TRIDENT.NUM_BRANCH = 3
