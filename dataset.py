import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes
from torchvision import transforms


class ObjectDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            split (str): One of 'train', 'valid', 'test'. Defines which subset of data to load.
            transform (callable, optional): Optional transform to be applied on an image and bounding box.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load COCO-style annotations
        annotations_path = os.path.join(root_dir, split, '_annotations.coco.json')
        with open(annotations_path, 'r') as f:
            coco = json.load(f)

        self.images_info = {img['file_name']: img for img in coco['images']}
        self.annotations_info = {}
        for ann in coco['annotations']:
            file_name = next((img['file_name'] for img in coco['images'] if img['id'] == ann['image_id']), None)
            if file_name:
                self.annotations_info[file_name] = {
                    "bbox": ann['bbox'],
                    "category_id": ann['category_id']
                }

        # Dynamically build class_to_idx from categories
        self.class_id_to_name = {cat['id']: cat['name'].lower() for cat in coco['categories']}
        self.class_to_idx = {name: i for i, name in enumerate(sorted(set(self.class_id_to_name.values())))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Collect image paths
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.image_paths = [
            os.path.join(self.image_dir, file_name)
            for file_name in self.images_info
            if file_name.endswith(('.jpg', '.png'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)

        # Convert PIL image to float tensor [0, 1]
        img = transforms.ToTensor()(img)  # shape: (3, H, W), dtype: float32

        metadata = self.annotations_info.get(filename, {})
        category_id = metadata.get("category_id", -1)
        cls_name = self.class_id_to_name.get(category_id, None)

        if cls_name is None or cls_name not in self.class_to_idx:
            raise ValueError(f"Unknown class id '{category_id}' for file '{filename}'.")

        label = self.class_to_idx[cls_name]

        bbox = metadata.get("bbox", [0, 0, 0, 0])
        bbox = torch.tensor(bbox, dtype=torch.float32)

        w, h = img.shape[2], img.shape[1]  # img is now (C, H, W)
        bbox_abs = BoundingBoxes(
            [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h],
            format="xywh",
            canvas_size=(h, w)
        )

        if self.transform:
            img, bbox_abs = self.transform(img, bbox_abs)

        bbox_abs = bbox_abs[0]

        bbox_norm = torch.tensor([
            bbox_abs[0] / w, bbox_abs[1] / h, bbox_abs[2] / w, bbox_abs[3] / h
        ], dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.long), bbox_norm