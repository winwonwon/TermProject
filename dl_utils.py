import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def train_one_epoch(
        dataloader, model, 
        class_loss_fn, bbox_loss_fn, 
        optimizer, scheduler, epoch, 
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

    # Increase bbox loss weight to prioritize localization
    class_loss_weight = 1.0
    bbox_loss_weight = 10.0  # Higher weight for bbox regression
    

    for i, (X, y, bboxes) in enumerate(dataloader):
        X, y, bboxes = X.to(device), y.to(device), bboxes.to(device)

        optimizer.zero_grad()

        # Model prediction
        class_pred, bbox_pred = model(X)

        # TODO: Compute classification and localization loss
        class_loss = class_loss_fn(class_pred, y) # YOUR CODE HERE
        # In test() function
        bbox_loss = bbox_loss_fn(bbox_pred, bboxes) # Remove incorrect 'y' argument
        # TODO: Computer the total loss
        total_loss = (class_loss_weight * class_loss) + (bbox_loss_weight * bbox_loss)

        # TODO: Backpropagation - which loss term you should use to run `backward()`?
        # YOUR CODE HERE
        total_loss.backward()
        
        # Inside train_one_epoch():
        optimizer.step()
        scheduler.step()  # Add this line

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
        class_loss (float), bbox_loss (float), avg_iou (float), 
        y_preds (tensor), y_trues (tensor), bbox_preds (tensor), bbox_trues (tensor)
    """
    num_batches = len(dataloader)
    model.eval()

    class_loss = 0
    bbox_loss = 0
    y_preds, y_trues = [], []
    bbox_preds, bbox_trues = [], []
    iou_scores = []

    def _calculate_iou(pred_boxes, true_boxes):
        """Calculate Intersection over Union (IoU) between predicted and true boxes"""
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        pred_boxes = pred_boxes.clone().detach()
        true_boxes = true_boxes.clone().detach()
        
        pred_boxes[:, 2:] = pred_boxes[:, :2] + pred_boxes[:, 2:]
        true_boxes[:, 2:] = true_boxes[:, :2] + true_boxes[:, 2:]
        
        # Intersection coordinates
        inter_x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
        
        # Intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        union_area = pred_area + true_area - inter_area
        
        return inter_area / (union_area + 1e-6)  # IoU for each sample in batch

    with torch.no_grad():
        for X, y, bboxes in dataloader:
            X, y, bboxes = X.to(device), y.to(device), bboxes.to(device)
            class_pred, bbox_pred = model(X)

            # Update loss calculation
            class_loss += class_loss_fn(class_pred, y).item()
            bbox_loss += bbox_loss_fn(bbox_pred, bboxes).item()  

            # Metrics collection
            y_preds.append(class_pred.argmax(1))
            y_trues.append(y.argmax(dim=1))
            bbox_preds.append(bbox_pred)
            bbox_trues.append(bboxes)
            
            # IoU calculation
            iou = _calculate_iou(bbox_pred.view(-1, 4), bboxes.view(-1, 4))
            iou_scores.append(iou)

    avg_iou = torch.cat(iou_scores).mean().item()
    return (
        class_loss / num_batches, 
        bbox_loss / num_batches,
        avg_iou,
        torch.cat(y_preds), 
        torch.cat(y_trues), 
        torch.cat(bbox_preds), 
        torch.cat(bbox_trues)
    )

def plot_predictions(
    images, labels, bboxes_true, class_names, 
    preds=None, bboxes_pred=None, num_samples=6, 
    save_path="predictions.jpg"
):
    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(20, 10))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_samples):
        img = images[i].detach()
        label_vec = labels[i].detach()  # Multi-label vector
        bbox_t = bboxes_true[i].detach().numpy()  # [num_boxes, 4]

        # Unnormalize image
        img = img * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

        ax = axes[i]
        ax.imshow(img)

        # Get class names for all detected classes (where label == 1)
        detected_classes = torch.where(label_vec == 1)[0]
        gt_label = ", ".join([class_names.get(c.item(), f"Class {c}") for c in detected_classes])

        # Draw all ground truth bboxes
        img_h, img_w, _ = img.shape
        for box in bbox_t:
            _draw_bbox(ax, box, img_h, img_w, "green", label=f"GT: {gt_label}")

        # Draw predicted bbox(es)
        if preds is not None:
            pred_vec = preds[i].detach()  # Multi-label predictions
            bbox_p = bboxes_pred[i].detach().numpy() if bboxes_pred is not None else None

            pred_classes = torch.where(pred_vec > 0.5)[0]  # Threshold at 0.5
            pred_label = ", ".join([class_names.get(c.item(), f"Class {c}") for c in pred_classes])

            if bbox_p is not None:
                for box in bbox_p:
                    _draw_bbox(ax, box, img_h, img_w, "red", label=f"Pred: {pred_label}")

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

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred/target format: [x, y, w, h] normalized
        # Convert to [x1, y1, x2, y2]
        pred = self._convert_boxes(pred)
        target = self._convert_boxes(target)

        # Calculate IoU
        inter_area = (torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0])) * \
                     (torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1]))
        inter_area = torch.clamp(inter_area, min=0)

        union_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + \
                     (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]) - inter_area

        iou = inter_area / (union_area + 1e-6)
        return 1 - iou.mean()

    def _convert_boxes(self, boxes):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)