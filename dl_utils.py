import torch
import matplotlib.pyplot as plt
import numpy as np


def train_one_epoch(
        dataloader, model, 
        class_loss_fn, bbox_loss_fn, 
        optimizer, epoch, 
        device, writer,
        log_step_interval=50):
    """
    Train a multi-task model for one epoch.

    Args:
        dataloader: PyTorch DataLoader
        model: PyTorch model (classification + localization)
        class_loss_fn: Loss function for classification (CrossEntropy)
        bbox_loss_fn: Loss function for localization (MSE)
        optimizer: Optimizer (Adam, SGD, etc.)
        epoch: Current epoch number
        device: Device to train on (CPU/GPU)
        writer: TensorBoard writer
        log_step_interval: Logging interval
    """
    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.

    for i, (X, y, bboxes) in enumerate(dataloader):
        X, y, bboxes = X.to(device), y.to(device), bboxes.to(device)

        optimizer.zero_grad()

        # Model prediction
        class_pred, bbox_pred = model(X)

        # TODO: Compute classification and localization loss
        class_loss = class_loss_fn(class_pred, y) # YOUR CODE HERE
        bbox_loss = bbox_loss_fn(bbox_pred, bboxes) # YOUR CODE HERE

        # TODO: Computer the total loss
        total_loss = class_loss + bbox_loss# YOUR CODE HERE

        # TODO: Backpropagation - which loss term you should use to run `backward()`?
        # YOUR CODE HERE
        total_loss.backward()
        
        optimizer.step()

        running_loss += total_loss.item()
        if (i+1) % log_step_interval == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} (Class: {class_loss.item():.4f}, BBox: {bbox_loss.item():.4f})")
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(dataloader) + i)


def test(dataloader, model, class_loss_fn, bbox_loss_fn, device):
    """
    Evaluate a multi-task model on the test set.

    Args:
        dataloader: PyTorch DataLoader
        model: Multi-task model
        class_loss_fn: Classification loss function
        bbox_loss_fn: Bounding box loss function
        device: Device (CPU/GPU)

    Returns:
        class_loss (float), bbox_loss (float), y_preds (tensor), y_trues (tensor),
        bbox_preds (tensor), bbox_trues (tensor)
    """
    num_batches = len(dataloader)
    model.eval()

    class_loss = 0
    bbox_loss = 0
    y_preds, y_trues = [], []
    bbox_preds, bbox_trues = [], []

    with torch.no_grad():
        for X, y, bboxes in dataloader:
            X, y, bboxes = X.to(device), y.to(device), bboxes.to(device)
            class_pred, bbox_pred = model(X)

            class_loss += class_loss_fn(class_pred, y).item()
            bbox_loss += bbox_loss_fn(bbox_pred, bboxes).item()

            y_preds.append(class_pred.argmax(1))
            y_trues.append(y)
            bbox_preds.append(bbox_pred)
            bbox_trues.append(bboxes)

    return class_loss / num_batches, bbox_loss / num_batches, torch.cat(y_preds), torch.cat(y_trues), torch.cat(bbox_preds), torch.cat(bbox_trues)


def plot_predictions(
        images, labels, bboxes_true, class_names, 
        preds=None, bboxes_pred=None, num_samples=6, 
        save_path="predictions.jpg"
    ):
    """
    Plots images with ground truth & predicted bounding boxes + classification labels.

    Args:
        images (torch.Tensor): Batch of images (shape: [batch_size, C, H, W])
        labels (torch.Tensor): Ground truth class labels
        preds (torch.Tensor): Predicted class labels
        bboxes_true (torch.Tensor): Ground truth bounding boxes (normalized, [x, y, w, h])
        bboxes_pred (torch.Tensor): Predicted bounding boxes (normalized, [x, y, w, h])
        class_names (dict): Dictionary mapping class indices to class names
        num_samples (int): Number of samples to display
        save_path (str): Path to save the plot
    """
    num_samples = min(num_samples, images.shape[0])  # Ensure we don't exceed batch size

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(20, 10))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_samples):
        img = images[i].detach()
        label = labels[i].item()
        bbox_t = bboxes_true[i].detach().numpy()  # Ground truth bbox

        # Unnormalize image
        img = img * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

        ax = axes[i]
        ax.imshow(img)
        
        # Draw ground truth bbox
        img_h, img_w, _ = img.shape
        # print(img.shape, img_w, img_h)
        _draw_bbox(ax, bbox_t, img_h, img_w, "green", label=f"GT: {class_names[label]}")
        
        # Draw predicted bbox
        if preds is not None:
            pred = preds[i].item()
            bbox_p = bboxes_pred[i].detach().numpy()  # Predicted bbox
            _draw_bbox(ax, bbox_p, img_h, img_w, "red", label=f"Pred: {class_names[pred]}")

        ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path)
    plt.close('all')


def _draw_bbox(ax, bbox, img_h, img_w, color, label=""):
    """
    Draws a bounding box on the given matplotlib axis.

    Args:
        ax: Matplotlib axis.
        bbox: Bounding box coordinates [x, y, w, h] (normalized 0-1).
        img_h: Image height.
        img_w: Image width.
        color: Color of the box.
        label: Label text.
    """
    x, y, w, h = bbox
    x *= img_w
    y *= img_h
    w *= img_w
    h *= img_h

    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(x, y - 2, label, color=color, fontsize=10, bbox=dict(facecolor="white", alpha=0.6))