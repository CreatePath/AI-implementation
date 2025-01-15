from yacs.config import CfgNode as CN

RESNET18 = CN()
RESNET18.BLOCK = "BASIC"
RESNET18.NUM_BLOCKS = [2, 2, 2, 2]
RESNET18.CHANNELS = [64, 128, 256, 512]

RESNET34 = CN()
RESNET34.BLOCK = "BASIC"
RESNET34.NUM_BLOCKS = [3, 4, 6, 3]
RESNET34.CHANNELS = [64, 128, 256, 512]

RESNET50 = CN()
RESNET50.BLOCK = "BOTTLENECK"
RESNET50.NUM_BLOCKS = [3, 4, 6, 3]
RESNET50.CHANNELS = [64, 128, 256, 512]

RESNET101 = CN()
RESNET101.BLOCK = "BOTTLENECK"
RESNET101.NUM_BLOCKS = [3, 4, 23, 3]
RESNET101.CHANNELS = [64, 128, 256, 512]

RESNET152 = CN()
RESNET152.BLOCK = "BOTTLENECK"
RESNET152.NUM_BLOCKS = [3, 8, 36, 3]
RESNET152.CHANNELS = [64, 128, 256, 512]

cfg = {"resnet18": RESNET18,
       "resnet34": RESNET34,
       "resnet50": RESNET50,
       "resnet101": RESNET101,
       "resnet152": RESNET152,}

def get_resnet_config(version: str):
    if version not in cfg.keys():
        raise ValueError(f"version({version}) is invalid. Please choose one of following resnet versions: {list(cfg.keys())}")
    return cfg[version]