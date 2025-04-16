import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import densenet121

class MultiTaskModel(nn.Module):
    # def __init__(self, num_classes=3):
    #     super(MultiTaskModel, self).__init__()
        
    #     # Load ResNet backbone
    #     weights = ResNet18_Weights.DEFAULT
    #     resnet = resnet18(weights=weights)
    #     self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fully connected layer

    #     # The output size of from self.backbone
    #     n_features = resnet.fc.in_features

    #     # TODO: Classification head
    #     self.classifier = nn.Linear(n_features,num_classes)

    #     # TODO: Localization head (bounding box regression)
    #     self.regressor = nn.Linear(n_features,4) #for x,y,w,h

    # def forward(self, x):
    #     out = self.backbone(x)
    #     out = torch.flatten(out, 1)  # Flatten the output

    #     # TODO: Model output
    #     class_out = self.classifier(out)
    #     bbox_out = self.regressor(out)

    #     return class_out, bbox_out
    def __init__(self, num_classes=3):
        super(MultiTaskModel, self).__init__()
        
        # Load ResNet backbone
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fully connected layer

        # The output size of from self.backbone
        n_features = resnet.fc.in_features

        # TODO: Classification head
        self.classifier = nn.Linear(n_features,num_classes)

        # TODO: Localization head (bounding box regression)
        self.regressor = nn.Linear(n_features,4) #for x,y,w,h

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)  # Flatten the output

        # TODO: Model output
        class_out = self.classifier(out)
        bbox_out = self.regressor(out)

        return class_out, bbox_out
    
class DenseNetMultiTaskModel(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNetMultiTaskModel, self).__init__()
        
        # Load DenseNet backbone with pretrained weights
        densenet = densenet121(pretrained=True)
        # Remove the classifier. DenseNet's features are accessed via densenet.features.
        self.backbone = densenet.features  
        # DenseNet's classifier input features are stored in densenet.classifier.in_features
        n_features = densenet.classifier.in_features

        # We need to add a pooling layer because densenet.features returns a feature map.
        # Global average pooling converts (N, n_features, H, W) to (N, n_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Linear(n_features, num_classes)
        # Bounding box regression head (output: x, y, w, h)
        self.regressor = nn.Linear(n_features, 4)

    def forward(self, x):
        # Extract features (shape: [N, n_features, H, W])
        features = self.backbone(x)
        # Apply global average pooling to get shape: [N, n_features, 1, 1]
        pooled = self.pool(features)
        # Flatten: shape becomes [N, n_features]
        out = pooled.view(pooled.size(0), -1)

        class_out = self.classifier(out)
        bbox_out = self.regressor(out)
        return class_out, bbox_out

class SimpleCNNMultiTaskModel(nn.Module):
    def __init__(self, num_classes=3, input_size=224):
        super(SimpleCNNMultiTaskModel, self).__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [N, 3, 224, 224] -> [N, 32, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> [N, 32, 112, 112]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [N, 64, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> [N, 64, 56, 56]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> [N, 128, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # -> [N, 128, 28, 28]
        )
        
        # Dynamically compute the flattened feature size
        self.flattened_size = self._get_conv_output(input_size)
        
        # Fully connected layers for shared features
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
        )
        
        # Classification head
        self.classifier = nn.Linear(512, num_classes)
        # Bounding box regression head (predict x, y, w, h)
        self.regressor = nn.Linear(512, 4)

    def _get_conv_output(self, input_size):
        # Create a dummy input to pass through conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            output = self.conv_layers(dummy_input)
            return int(torch.flatten(output, 1).shape[1])
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_shared(x)
        
        class_out = self.classifier(x)
        bbox_out = self.regressor(x)
        return class_out, bbox_out

