###########################
# Import Python Packages
###########################
import os
import json
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms_v2
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
import torch.nn.functional as F

from dl_utils import train_one_epoch, test, plot_predictions
from model import MultiTaskModel, SimpleCNNMultiTaskModel, DenseNetMultiTaskModel
from dataset import ObjectDataset

####################
# Hyperparameters
####################
learning_rate = 1e-5    # TODO: Change here as you see fit
batch_size = 8          # TODO: Change here as you see fit
epochs = 10              # TODO: Change here as you see fit


####################
# Dataset
####################
DATA_DIR = "dataset"
LABELS_FILE = os.path.join(DATA_DIR, "train/_annotations.coco.json")

# Load labels
with open(LABELS_FILE, "r") as f:
    labels = json.load(f)

# Define Augmentations for Training
train_transforms = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    # transforms_v2.RandomHorizontalFlip(p=0.5),
    # transforms_v2.RandomVerticalFlip(p=0.2),
    # transforms_v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    # transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # transforms_v2.GaussianBlur(kernel_size=3),
    #transforms_v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms_v2.ToImage(),
    transforms_v2.Grayscale(num_output_channels=1),             # Convert to grayscale (B&W)
    transforms_v2.Lambda(lambda x: x.repeat(3, 1, 1)),          # Repeat channel to convert back to 3-channel for model input
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # Define Transformations for Validation & Test (No Augmentation, but with Bounding Box Support)
test_transforms = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToImage(),
    transforms_v2.Grayscale(num_output_channels=1),             # Convert to grayscale (B&W)
    transforms_v2.Lambda(lambda x: x.repeat(3, 1, 1)),          # Repeat channel to convert back to 3-channel for model input
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ObjectDataset(root_dir="dataset", split="train")

# Split into train, valid, and test
torch.manual_seed(42)
n_valid_samples = int(0.1 * len(full_dataset))
n_test_samples = int(0.1 * len(full_dataset))
n_train_samples = len(full_dataset) - n_valid_samples - n_test_samples


# train_ds, valid_ds, test_ds = random_split(full_dataset, [n_train_samples, n_valid_samples, n_test_samples])

train_ds = ObjectDataset(DATA_DIR, split="train",transform=train_transforms)
valid_ds = ObjectDataset(DATA_DIR, split="valid",transform=test_transforms)
test_ds = ObjectDataset(DATA_DIR, split="test",transform=test_transforms)

# # Apply transformations
# train_ds.dataset.transform = train_transforms
# valid_ds.dataset.transform = test_transforms
# test_ds.dataset.transform = test_transforms

# Define DataLoaders
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Plot dataset
_images, _labels, _bboxes = next(iter(train_dl))
plot_predictions(
    _images, _labels, _bboxes, full_dataset.idx_to_class,
    num_samples=8, save_path="data.jpg",
)


####################
# Model
####################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# TODO: Create a model
# Get actual number of classes from dataset
num_classes = len(train_ds.class_to_idx)

# Update model initialization
model = MultiTaskModel(num_classes=num_classes, max_cracks=10).to(device)
#model = SimpleCNNMultiTaskModel().to(device)
#model = DenseNetMultiTaskModel().to(device)
print(model)


####################
# Model Training
####################
writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

# TODO: Define loss functions
class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.MSELoss()

# TODO: Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop
best_vloss = float('inf')
for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")

    train_one_epoch(
        dataloader=train_dl, 
        model=model, 
        class_loss_fn=class_loss_fn, 
        bbox_loss_fn=bbox_loss_fn, 
        optimizer=optimizer, 
        scheduler=scheduler,
        epoch=epoch, 
        device=device, 
        writer=writer,
        log_step_interval=1,
    )

    # Compute train & validation loss
    train_loss, train_bbox_loss, train_iou, train_y_preds, train_y_trues, train_bbox_preds, train_bbox_trues = test(
        train_dl, model, class_loss_fn, bbox_loss_fn, device
    )
    # Use VALIDATION SET for final metrics
    val_loss, val_bbox_loss, val_iou, val_y_preds, val_y_trues, val_bbox_preds, val_bbox_trues = test(
        valid_dl, model, class_loss_fn, bbox_loss_fn, device
    )

    # Compute classification metrics
    if val_y_trues.ndim == 2:
        val_y_trues = val_y_trues.argmax(dim=1)
    if train_y_trues.ndim == 2:
        train_y_trues = train_y_trues.argmax(dim=1)

    train_accuracy = multiclass_accuracy(train_y_preds, train_y_trues).item()
    train_f1 = multiclass_f1_score(train_y_preds, train_y_trues, num_classes=num_classes).item()
    val_accuracy = multiclass_accuracy(val_y_preds, val_y_trues, num_classes=num_classes).item()
    val_f1 = multiclass_f1_score(val_y_preds, val_y_trues, num_classes=num_classes).item()

    # ✅ Debug print for sanity check
    print("\n[DEBUG] Sample Predictions vs Ground Truth (val):")
    print("val_y_preds[:10]:", val_y_preds[:10])
    print("val_y_trues[:10]:", val_y_trues[:10])
    print("Correct predictions:", (val_y_preds == val_y_trues).sum().item(), "/", len(val_y_preds))

    # Compute bounding box MSE
    train_bbox_mse = F.mse_loss(train_bbox_preds, train_bbox_trues).item()
    val_bbox_mse = F.mse_loss(val_bbox_preds, val_bbox_trues).item()

    # Log training performance (add IoU)
    writer.add_scalars('Train vs. Valid/loss', 
        {'train': train_loss, 'valid': val_loss}, 
        epoch)
    writer.add_scalars('Train vs. Valid/bbox_mse', 
        {'train': train_bbox_mse, 'valid': val_bbox_mse}, 
        epoch)
    writer.add_scalars('Train vs. Valid/iou', 
        {'train': train_iou, 'valid': val_iou},  # NEW: Log IoU
        epoch)
    writer.add_scalars('Train vs. Valid/acc', 
        {'train': train_accuracy, 'valid': val_accuracy}, 
        epoch)
    writer.add_scalars('Train vs. Valid/f1', 
        {'train': train_f1, 'valid': val_f1}, 
        epoch)

    # Save the best model
    if val_loss < best_vloss:
        best_vloss = val_loss
        torch.save(model.state_dict(), 'model_best_vloss.pth')
        print('Saved best model to model_best_vloss.pth')

###########################
# Evaluate on the Test Set
###########################
# TODO: Load the best model
model = model.to(device)

# Evaluate on the test set
test_loss, test_bbox_loss, test_iou, test_y_preds, test_y_trues, test_bbox_preds, test_bbox_trues = test(
    test_dl, model, class_loss_fn, bbox_loss_fn, device
)

# If needed, convert test labels to class index
if test_y_trues.ndim == 2:
    test_y_trues = test_y_trues.argmax(dim=1)

# Compute test classification metrics
test_accuracy = multiclass_accuracy(test_y_preds, test_y_trues).item()
test_f1 = multiclass_f1_score(test_y_preds, test_y_trues).item()

print("\n[DEBUG] Sample Predictions vs Ground Truth (test):")
print("test_y_preds[:10]:", test_y_preds[:10])
print("test_y_trues[:10]:", test_y_trues[:10])
print("Correct predictions:", (test_y_preds == test_y_trues).sum().item(), "/", len(test_y_preds))

# Compute bounding box MSE
test_bbox_mse = F.mse_loss(test_bbox_preds, test_bbox_trues).item()

print(f"\nTest Results:")
print(f"Classification Loss: {test_loss:.4f}")
print(f"Bounding Box MSE: {test_bbox_mse:.4f}")
print(f"Bounding Box IoU: {test_iou:.4f}")  # NEW: Print IoU
print(f"Accuracy: {test_accuracy:.2f}%")
print(f"F1 Score: {test_f1:.2f}")

# Get a batch from test set
test_images, test_labels, test_bboxes = next(iter(test_dl))

# TODO: Make predictions
test_preds, test_bboxes_pred = model(test_images.to(device))
test_preds = test_preds.argmax(1)

# Plot predictions
plot_predictions(
    test_images, test_labels, test_bboxes, full_dataset.idx_to_class,
    test_preds.cpu(), test_bboxes_pred.cpu(), 
    num_samples=8, save_path="predictions.jpg",
)