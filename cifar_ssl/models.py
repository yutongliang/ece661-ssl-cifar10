"""Encoders shared by every method (SimCLR, RotNet, SimSiam, supervised).

resnet20: He et al. 2016, Sec. 4.2 (the HW2 model): 3 stages of 3 basic blocks
    with 16/32/64 channels and zero-padding (option A) shortcuts; 64-d feature.
resnet50: torchvision ResNet-50 with the SimCLR Appendix B.9 CIFAR stem (3x3
    stride-1 first conv, no max-pool); 2048-d feature.

Each encoder stops at global average pooling. Each method adds its own head on
top (linear classifier, rotation head, projection MLP, ...).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

ARCHITECTURES = ("resnet20", "resnet50")


class ZeroPadShortcut(nn.Module):
    """Option A shortcut: subsample spatially and zero-pad the extra channels."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.channel_padding = out_channels - in_channels

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.channel_padding > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_padding))
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ZeroPadShortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """ResNet-20 without the final classifier.

    forward(x)                      -> (B, 64) pooled features
    forward(x, return_stages=True)  -> list of the 3 stage feature maps
                                       [(B,16,32,32), (B,32,16,16), (B,64,8,8)]
    """

    feature_dim = 64
    stage_channels = (16, 32, 64)

    def __init__(self, blocks_per_stage=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        stages = []
        in_channels = 16
        for i, out_channels in enumerate(self.stage_channels):
            blocks = []
            for j in range(blocks_per_stage):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_stages=False):
        x = F.relu(self.bn1(self.conv1(x)))
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)
        if return_stages:
            return stage_outputs
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class ResNet50CIFAR(nn.Module):
    """ResNet-50 with the SimCLR B.9 CIFAR-10 stem and no classifier.

    forward(x)                      -> (B, 2048) pooled features
    forward(x, return_stages=True)  -> the 4 stage feature maps
                                       [(B,256,32,32), (B,512,16,16), (B,1024,8,8), (B,2048,4,4)]
    """

    feature_dim = 2048

    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)
        # B.9: replace the 7x7 stride-2 conv with a 3x3 stride-1 conv; remove the max-pool.
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

    def forward(self, x, return_stages=False):
        x = self.stem(x)
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)
        if return_stages:
            return stage_outputs
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def build_encoder(arch="resnet20"):
    if arch == "resnet20":
        return ResNet20()
    if arch == "resnet50":
        return ResNet50CIFAR()
    raise ValueError(f"Unknown architecture {arch!r}; choose from {ARCHITECTURES}")


def checkpoint_arch(ckpt):
    """Architecture recorded in an encoder checkpoint; older checkpoints are ResNet-20."""
    return ckpt.get("args", {}).get("arch", "resnet20")


class Classifier(nn.Module):
    """Encoder + linear layer. Used for the supervised baseline and fine-tuning."""

    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


def load_encoder(checkpoint_path, device="cpu"):
    """Load an encoder saved by any pretraining script (see utils.save_encoder)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder = build_encoder(checkpoint_arch(ckpt))
    encoder.load_state_dict(ckpt["encoder"])
    return encoder, ckpt
