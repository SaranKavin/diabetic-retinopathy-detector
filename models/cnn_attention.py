import torch.nn as nn
import torchvision.models as models

class CNNAttention(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNAttention, self).__init__()
        base = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
