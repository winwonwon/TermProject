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
        img = Image.open(image_path).convert("RGB")  # Opening as PIL Image
        filename = os.path.basename(image_path)

        img_h, img_w = img.size[1], img.size[0]  # height, width

        annotations = self.annotations_info.get(filename, [])
        bboxes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            x /= img_w
            y /= img_h
            w /= img_w
            h /= img_h
            bboxes.append([x, y, w, h])
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4))
        max_boxes = 10
        if len(bboxes) < max_boxes:
            pad = torch.zeros((max_boxes - len(bboxes), 4))
            bboxes = torch.cat([bboxes, pad])
        else:
            bboxes = bboxes[:max_boxes]

        if self.transform:
            img = self.transform(img)  # this transform now converts the image to float

        return img, bboxes

    def __len__(self):
        return len(self.image_paths)  # Critical for DataLoader to work