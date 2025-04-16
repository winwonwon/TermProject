 ###########################
# Import Python Packages
###########################
import os
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms_v2
import torch.nn.functional as F
from torchvision import transforms

from dl_utils2 import train_one_epoch, test, plot_predictions
from model2 import BBoxOnlyModel, BBoxOnlyDenseNet121Model, BBoxEfficientNetModel
from dataset2 import ObjectDataset

####################
# Hyperparameters
####################
learning_rate = 1e-4
batch_size = 8
epochs = 3

####################
# Dataset & Transforms
####################
DATA_DIR = "dataset"

# train_transforms = transforms_v2.Compose([
#     transforms_v2.Resize((224, 224)),
#     transforms_v2.ToImage(),
#     transforms_v2.Grayscale(num_output_channels=1),
#     transforms_v2.Lambda(lambda x: x.repeat(3, 1, 1)),
#     #transforms_v2.ToDtype(torch.float32, scale=True),
#     transforms_v2.ToTensor(),
#     transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),     # Random crop and resize
    transforms.RandomHorizontalFlip(),            # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),                        # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

test_transforms = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToImage(),
    transforms_v2.Grayscale(num_output_channels=1),
    transforms_v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms_v2.ToTensor(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = ObjectDataset(DATA_DIR, split="train", transform=train_transforms)
valid_ds = ObjectDataset(DATA_DIR, split="valid", transform=test_transforms)
test_ds = ObjectDataset(DATA_DIR, split="test", transform=test_transforms)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

####################
# Model Setup
####################
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BBoxOnlyModel(max_cracks=10).to(device)
#model = BBoxEfficientNetModel(max_cracks=10).to(device)
print("Model initialized on", device)

####################
# Training Setup
####################
writer = SummaryWriter(f'./runs/bbox_only_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
bbox_loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_vloss = float('inf')

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_one_epoch(
        dataloader=train_dl,
        model=model,
        bbox_loss_fn=bbox_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        device=device,
        writer=writer,
        log_step_interval=10
    )

    train_loss, train_iou, train_bbox_preds, train_bbox_trues = test(
        train_dl, model, bbox_loss_fn, device
    )

    val_loss, val_iou, val_bbox_preds, val_bbox_trues = test(
        valid_dl, model, bbox_loss_fn, device
    )


    print(f"[Train] BBox MSE: {train_loss:.4f} | IoU: {train_iou:.4f}")
    print(f"[Valid] BBox MSE: {val_loss:.4f} | IoU: {val_iou:.4f}")

    writer.add_scalars("BBox MSE", {"train": train_loss, "valid": val_loss}, epoch)
    writer.add_scalars("IoU", {"train": train_iou, "valid": val_iou}, epoch)

    if val_loss < best_vloss:
        best_vloss = val_loss
        torch.save(model.state_dict(), "best_bbox_model.pth")
        print("Best model saved.")

###########################
# Evaluate on the Test Set
###########################
model.load_state_dict(torch.load("best_bbox_model.pth",weights_only=True))
model.eval()

test_loss, test_iou, test_bbox_preds, test_bbox_trues = test(
        test_dl, model, bbox_loss_fn, device
)

print(f"\nTest Results:")
print(f"BBox MSE: {test_loss:.4f}")
print(f"IoU: {test_iou:.4f}")

# Visualize Predictions
test_images, test_bboxes = next(iter(test_dl))
with torch.no_grad():
    test_bboxes_pred = model(test_images.to(device)).cpu()

plot_predictions(
    test_images, test_bboxes, bboxes_pred=test_bboxes_pred,
    num_samples=8, save_path="bbox_predictions.jpg"
)
