from torch import nn
from seresnet import seresnet18
import torch
from torch.nn import functional as F
import math
from config_ import config


'''
class ArcFace(nn.Module):
    def __init__(self, scale_factor=64.0, margin=0.5) -> None:
        super().__init__()
        self.scale = scale_factor
        self.margin = margin
    
    def forward(self, logits: torch.Tensor):
        logits = F.normalize(logits, p=2, dim=1)
        logits = logits.arccos() + self.margin
        logits = logits.cos()
        
        return logits * self.scale
'''


class ArcFace(nn.Module):
    def __init__(self, n_classes=1, in_features=512, scale_factor=64.0, margin=0.5) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features).to(config.device))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
    
    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).squeeze(-1)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        coeff = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        coeff = torch.where(cosine > -self.cos_m, coeff, cosine - self.margin * self.sin_m).squeeze(-1)

        output = labels * coeff + (1.0 - labels) * cosine
        return output * self.scale_factor


class ArcFaceLoss(nn.modules.loss._Loss):
    def __init__(self) -> None:
        super().__init__()

        self.arcface = ArcFace()
        weight = torch.Tensor([100.0]).to(config.device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, labels: torch.Tensor):
        logits = (embedding1 - embedding2).pow(2)
        return self.bce(self.arcface(logits, labels), labels)


class Baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.feature_extractor = seresnet18()
    
    def forward(self, x):
        return self.feature_extractor.forward_features(x)


class ContrastiveSigmoidWrapper(nn.modules.loss._Loss):
    def __init__(self, bias=1.0) -> None:
        super().__init__()
        self.bias = bias
        weight = torch.Tensor([100.0]).to(config.device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, labels):
        score = (embedding1 - embedding2).pow(2).sum(1) - self.bias

        return self.bce(score, labels)
