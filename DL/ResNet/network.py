import torch
from torch import nn

def init_weights(submodule):
    if isinstance(submodule, nn.Conv2d) or isinstance(submodule, nn.Linear):
        nn.init.kaiming_normal_(submodule.weight)
        submodule.bias.data.fill_(0.)
    elif isinstance(submodule, nn.BatchNorm2d):
        submodule.weight.data.fill_(1.)
        submodule.bias.data.fill_(0.)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.activation(out)

        return out


BLOCKS = {"BASIC": BasicBlock,
          "BOTTLENECK": Bottleneck,}


class ResNet(nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super().__init__()
        block = BLOCKS[cfg.BLOCK]

        self.conv1 = nn.Conv2d(3, cfg.CHANNELS[0], 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(cfg.CHANNELS[0])
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.in_channels = cfg.CHANNELS[0]
        self.layer1 = self._make_layer(block, cfg.NUM_BLOCKS[0], self.in_channels, cfg.CHANNELS[0], 1)
        self.layer2 = self._make_layer(block, cfg.NUM_BLOCKS[1], self.in_channels, cfg.CHANNELS[1], 2)
        self.layer3 = self._make_layer(block, cfg.NUM_BLOCKS[2], self.in_channels, cfg.CHANNELS[2], 2)
        self.layer4 = self._make_layer(block, cfg.NUM_BLOCKS[3], self.in_channels, cfg.CHANNELS[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg.CHANNELS[-1] * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            m.apply(init_weights)

    def _make_layer(self, block, num_block, in_channels, out_channels, stride):
        downsample = None
        if in_channels != out_channels * block.expansion or stride != 1:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels * block.expansion, 1, stride),
                                       nn.BatchNorm2d(out_channels * block.expansion),)

        layer = nn.ModuleList([block(in_channels, out_channels, stride, downsample)])
        self.in_channels = out_channels * block.expansion

        for _ in range(num_block-1):
            layer.append(block(self.in_channels, out_channels, 1))

        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.softmax(x)

        return x
