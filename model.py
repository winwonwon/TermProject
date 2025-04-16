import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=3, max_cracks=10):
        super().__init__()
        self.max_cracks = max_cracks

        # Load pretrained ResNet18 backbone
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC
        n_features = resnet.fc.in_features

        # Classification Head (Dropout + BatchNorm)
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),                   # Dropout for regularization
            nn.Linear(512, num_classes),
        )

        # Bounding Box Head (Dropout + BatchNorm)
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),                   # Dropout for regularization
            nn.Linear(512, max_cracks * 4),
        )

    def forward(self, x):
        features = self.backbone(x).flatten(1)  # Shape: [B, n_features]
        class_preds = self.classifier(features)
        bbox_preds = self.regressor(features).view(-1, self.max_cracks, 4)
        return class_preds, bbox_preds

class DenseNetMultiTaskModel(nn.Module):
    def __init__(self, num_classes=3, max_cracks=10):
        super().__init__()
        self.max_cracks = max_cracks
        densenet = densenet121(pretrained=True)
        self.backbone = densenet.features
        n_features = densenet.classifier.in_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_features, num_classes)
        self.regressor = nn.Linear(n_features, max_cracks * 4)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        class_out = self.classifier(pooled)
        bbox_out = self.regressor(pooled).view(-1, self.max_cracks, 4)
        return class_out, bbox_out

class SimpleCNNMultiTaskModel(nn.Module):
    def __init__(self, num_classes=3, input_size=224, max_cracks=10):
        super().__init__()
        self.max_cracks = max_cracks
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flattened_size = self._get_conv_output(input_size)
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(512, num_classes)
        self.regressor = nn.Linear(512, max_cracks * 4)

    def _get_conv_output(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            return torch.flatten(self.conv_layers(dummy_input), 1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x).flatten(1)
        x = self.fc_shared(x)
        class_out = self.classifier(x)
        bbox_out = self.regressor(x).view(-1, self.max_cracks, 4)
        return class_out, bbox_out