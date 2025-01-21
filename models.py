import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)


class ResNet_1x1(torch.nn.Module):
    def __init__(self, args):
        super(ResNet_1x1, self).__init__()
        feature_extractor = torchvision.models.resnet50(pretrained=args.use_pretrained)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])

        if args.freeze_feature_extractor:
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True

        self.feature_extractor = feature_extractor
        self.avgpool = GlobalAvgPool2d()
        self.onebyone_conv = nn.Conv2d(args.feat_dim, args.num_classes, 1)

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, x):
        feats = self.feature_extractor(x)
        CAM = self.onebyone_conv(feats)
        logits = F.adaptive_avg_pool2d(CAM, 1).squeeze(-1).squeeze(-1)
        return logits, CAM
