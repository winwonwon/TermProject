import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121
from efficientnet_pytorch import EfficientNet

class BBoxOnlyModel(nn.Module):
    def __init__(self, max_cracks=10):
        super().__init__()
        self.max_cracks = max_cracks

        # ResNet50 Backbone
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        n_features = resnet.fc.in_features

        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, max_cracks * 4)
        )

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        bbox_out = self.regressor(features).view(-1, self.max_cracks, 4)
        return bbox_out
    
class BBoxOnlyDenseNet121Model(nn.Module):
    def __init__(self, max_cracks=10):
        super().__init__()
        self.max_cracks = max_cracks

        # DenseNet121 Backbone
        densenet = densenet121(pretrained=True)
        self.backbone = nn.Sequential(*list(densenet.children())[:-1])  # Remove FC layer
        
        # Dynamically calculate the number of features from the backbone output
        self.backbone_sample_input = torch.zeros(1, 3, 224, 224)  # Assuming input image size is 224x224
        with torch.no_grad():
            sample_output = self.backbone(self.backbone_sample_input)
        n_features = sample_output.view(sample_output.size(0), -1).size(1)  # Flatten and get the feature size

        # Define the regressor network
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, max_cracks * 4)  # Output: max_cracks boxes with 4 coordinates each
        )

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x).flatten(1)
        # Predict bounding boxes using the regressor
        bbox_out = self.regressor(features).view(-1, self.max_cracks, 4)
        return bbox_out

class BBoxEfficientNetModel(nn.Module):
    def __init__(self, max_cracks=10, backbone='efficientnet-b0'):
        super(BBoxEfficientNetModel, self).__init__()
        self.max_cracks = max_cracks
        
        # Load pretrained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained(backbone)
        
        # Use the built-in method extract_features() for feature extraction
        # This returns a feature map with shape (batch, 1280, H, W)
        
        # Feature Pyramid Network (FPN)
        # We'll assume that after extract_features(), H and W are 7 (for 224x224 input)
        # and our FPN adjusts the channel dimension.
        self.fpn = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),  # From 1280 channels to 512
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1),   # From 512 to 256
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1)    # From 256 to 128
        )
        
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bounding box regressor using the pooled features.
        # Now the flattened feature vector will be of size 128.
        self.regressor = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, max_cracks * 4)
        )

    def forward(self, x):
        # Extract features from EfficientNet using its built-in method
        features = self.efficientnet.extract_features(x)  # shape: (batch, 1280, H, W)
        
        # Process features through the FPN
        features = self.fpn(features)  # shape becomes (batch, 128, H, W)
        
        # Adaptive pooling: reduce spatial dimensions to 1x1
        features = self.avgpool(features)  # shape: (batch, 128, 1, 1)
        
        # Flatten the features: becomes (batch, 128)
        features = features.view(features.size(0), -1)
        
        # Regress bounding boxes: output shape (batch, max_cracks*4) then reshaped
        bbox_out = self.regressor(features).view(-1, self.max_cracks, 4)
        
        return bbox_out
