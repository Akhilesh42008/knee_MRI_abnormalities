import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class MRNetClassifier(nn.Module):
    """Shared ResNet encoder with slice and view pooling."""

    def __init__(self, pretrained=False, dropout=0.4):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        # images: [batch, views, slices, channels, height, width]
        batch, views, slices, channels, height, width = images.shape
        images = images.reshape(batch * views * slices, channels, height, width)
        features = self.encoder(images).flatten(1)
        features = features.reshape(batch, views, slices, -1)
        features = features.max(dim=2).values
        features = features.reshape(batch, views * features.shape[-1])
        return self.classifier(features).squeeze(1)
