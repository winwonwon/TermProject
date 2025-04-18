import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes
from torchvision import transforms

class ObjectDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load COCO-style annotations
        annotations_path = os.path.join(root_dir, split, '_annotations.coco.json')
        with open(annotations_path, 'r') as f:
            self.coco = json.load(f)  # Store full COCO data

        # Map image filenames to their metadata
        self.images_info = {img['file_name']: img for img in self.coco['images']}

        # Group annotations by image filename
        self.annotations_info = {}
        for ann in self.coco['annotations']:
            file_name = next(
                (img['file_name'] for img in self.coco['images'] 
                if img['id'] == ann['image_id']), None)
            if file_name:
                if file_name not in self.annotations_info:
                    self.annotations_info[file_name] = []
                self.annotations_info[file_name].append(ann)  # List of annotations per image

        # Build class mappings
        self.class_id_to_name = {cat['id']: cat['name'].lower() for cat in self.coco['categories']}
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(self.class_id_to_name.values()))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Collect image paths
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.image_paths = [
            os.path.join(self.image_dir, f) for f in self.images_info 
            if f.endswith(('.jpg', '.png'))
        ]
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)

        # Convert PIL image to tensor
        img = transforms.ToTensor()(img)  # Shape: [3, H, W]
        img_h, img_w = img.shape[1:]      # Get height and width

        # Initialize single-label (multiclass) vector
        num_classes = len(self.class_to_idx)
        label = torch.zeros(num_classes, dtype=torch.float32)

        # Get annotations for this image
        annotations = self.annotations_info.get(filename, [])

        # Collect bounding boxes and classes
        bboxes = []
        for ann in annotations:
            class_name = self.class_id_to_name[ann['category_id']]
            class_idx = self.class_to_idx[class_name]
            label[class_idx] = 1.0  # Mark class as present

            # Normalize bbox to [0, 1]
            x, y, w, h = ann['bbox']
            x /= img_w
            y /= img_h
            w /= img_w
            h /= img_h
            bboxes.append([x, y, w, h])

        # Convert to tensor
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4))

        # Pad or truncate to fixed number of boxes
        max_boxes = 10
        if len(bboxes) < max_boxes:
            padding = torch.zeros((max_boxes - len(bboxes), 4), dtype=torch.float32)
            bboxes = torch.cat([bboxes, padding])
        else:
            bboxes = bboxes[:max_boxes]

        # Apply optional transforms
        if self.transform:
            img, bboxes = self.transform(img, bboxes)

        return img, label, bboxes

    
    def __len__(self):
        return len(self.image_paths)  # Critical for DataLoader to work