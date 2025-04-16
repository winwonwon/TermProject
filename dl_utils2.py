import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch(dataloader, model, bbox_loss_fn, optimizer, scheduler, epoch, device, writer, log_step_interval=50):
    model.train()
    running_loss = 0.0

    for i, (X, bboxes) in enumerate(dataloader):
        X, bboxes = X.to(device), bboxes.to(device)

        optimizer.zero_grad()
        bbox_pred = model(X)
        loss = bbox_loss_fn(bbox_pred, bboxes)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if (i + 1) % log_step_interval == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

def test(dataloader, model, bbox_loss_fn, device):
    model.eval()
    bbox_loss = 0
    bbox_preds, bbox_trues = [], []
    iou_scores = []

    def _calculate_iou(pred_boxes, true_boxes):
        pred_boxes = pred_boxes.clone().detach()
        true_boxes = true_boxes.clone().detach()
        pred_boxes[:, 2:] += pred_boxes[:, :2]
        true_boxes[:, 2:] += true_boxes[:, :2]

        inter_x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        union_area = pred_area + true_area - inter_area
        return inter_area / (union_area + 1e-6)

    with torch.no_grad():
        for X, bboxes in dataloader:
            X, bboxes = X.to(device), bboxes.to(device)
            bbox_pred = model(X)

            bbox_loss += bbox_loss_fn(bbox_pred, bboxes).item()
            bbox_preds.append(bbox_pred)
            bbox_trues.append(bboxes)

            iou = _calculate_iou(bbox_pred.view(-1, 4), bboxes.view(-1, 4))
            iou_scores.append(iou)

    avg_iou = torch.cat(iou_scores).mean().item()
    return (
        bbox_loss / len(dataloader),
        avg_iou,
        torch.cat(bbox_preds),
        torch.cat(bbox_trues)
    )

def plot_predictions(images, bboxes_true, bboxes_pred, num_samples=6, save_path="predictions2.jpg"):
    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(20, 10))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_samples):
        img = images[i].detach()
        bbox_t = bboxes_true[i].detach().numpy()
        bbox_p = bboxes_pred[i].detach().numpy()

        img = img * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        ax = axes[i]
        ax.imshow(img)

        img_h, img_w, _ = img.shape
        for box in bbox_t:
            _draw_bbox(ax, box, img_h, img_w, "green", label="GT")
        for box in bbox_p:
            _draw_bbox(ax, box, img_h, img_w, "red", label="Pred")

        ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path)
    plt.close('all')

def _draw_bbox(ax, bbox, img_h, img_w, color, label=""):
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
        pred = self._convert_boxes(pred)
        target = self._convert_boxes(target)

        inter_area = (torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0])) * \
                     (torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1]))
        inter_area = torch.clamp(inter_area, min=0)

        union_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + \
                     (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]) - inter_area

        iou = inter_area / (union_area + 1e-6)
        return 1 - iou.mean()

    def _convert_boxes(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)
    
class GIoULoss(nn.Module):
    def __init__(self):
        super(GIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = self._convert_boxes(pred)
        target = self._convert_boxes(target)

        inter_area = (torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0])) * \
                     (torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1]))
        inter_area = torch.clamp(inter_area, min=0)

        union_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + \
                     (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]) - inter_area

        iou = inter_area / (union_area + 1e-6)

        # Calculate the enclosing box area (the smallest box that contains both predicted and target boxes)
        enclose_area = (torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0])) * \
                       (torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1]))

        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
        return 1 - giou.mean()

    def _convert_boxes(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)
