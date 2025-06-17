# IOAI Computer Vision Competition Solutions
# Основные сценарии: Object Detection, Instance Segmentation, Semantic Segmentation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import json
from sklearn.metrics import average_precision_score
from transformers import AutoFeatureExtractor, AutoModel
import matplotlib.pyplot as plt

# =============================================================================
# СЦЕНАРИЙ 1: YOLO Object Detection
# =============================================================================

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx]['image_id']}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Предполагаем формат: image_id, x1, y1, x2, y2, class_id
        bbox = torch.tensor([
            self.data.iloc[idx]['x1'], self.data.iloc[idx]['y1'],
            self.data.iloc[idx]['x2'], self.data.iloc[idx]['y2']
        ], dtype=torch.float32)
        
        class_id = torch.tensor(self.data.iloc[idx]['class_id'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, bbox, class_id

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=80, grid_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 1024)
        
        # YOLO head: 5 bbox params + num_classes
        self.detector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, grid_size * grid_size * (5 + num_classes))
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.detector(features)
        return output.view(-1, self.grid_size, self.grid_size, 5 + self.num_classes)

def train_yolo(model, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_bbox = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, bboxes, classes in train_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            classes = classes.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Упрощенная loss функция
            bbox_pred = outputs[:, 0, 0, :4]  # Берем первую ячейку
            cls_pred = outputs[:, 0, 0, 5:]
            
            bbox_loss = criterion_bbox(bbox_pred, bboxes)
            cls_loss = criterion_cls(cls_pred, classes)
            
            loss = bbox_loss + cls_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# =============================================================================
# СЦЕНАРИЙ 2: Fast R-CNN
# =============================================================================

class FastRCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.backbone.fc = nn.Identity()
        
        self.roi_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )
        
    def forward(self, x, rois):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # ROI pooling (упрощенная версия)
        pooled_features = self.roi_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        cls_scores = self.classifier(pooled_features)
        bbox_deltas = self.bbox_regressor(pooled_features)
        
        return cls_scores, bbox_deltas

# =============================================================================
# СЦЕНАРИЙ 3: Mask R-CNN для Instance Segmentation
# =============================================================================

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # FPN-like structure
        self.fpn = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.ReLU()
        )
        
        # Detection head
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4, 1)
        )
        
        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # FPN
        features = self.fpn(x)
        
        # Heads
        cls_output = self.cls_head(features)
        bbox_output = self.bbox_head(features)
        mask_output = self.mask_head(features)
        
        return cls_output, bbox_output, mask_output

# =============================================================================
# СЦЕНАРИЙ 4: DeepLab для Semantic Segmentation
# =============================================================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)
        
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.global_pool(x)
        x5 = self.conv5(x5)
        x5 = nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.final_conv(out)

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        
        # Удаляем stride в последних слоях для dense prediction
        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        
        self.aspp = ASPP(2048, 256)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        input_shape = x.shape[2:]
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.aspp(x)
        x = self.classifier(x)
        
        # Upsampling to original size
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

# =============================================================================
# ОБЩИЕ УТИЛИТЫ ДЛЯ ТРЕНИРОВКИ И ИНФЕРЕНСА
# =============================================================================

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).squeeze(0).long()
            
        return image, mask

def train_segmentation_model(model, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

def create_submission(model, test_loader, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            # Convert predictions to required format
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for j, pred in enumerate(preds):
                predictions.append({
                    'image_id': image_ids[j],
                    'prediction': pred.tolist()
                })
    
    # Save predictions
    pd.DataFrame(predictions).to_csv(output_file, index=False)

# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

def main():
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Для YOLO
    yolo_model = SimpleYOLO(num_classes=80)
    
    # Для Fast R-CNN
    fast_rcnn_model = FastRCNN(num_classes=80)
    
    # Для Mask R-CNN
    mask_rcnn_model = MaskRCNN(num_classes=80)
    
    # Для DeepLab
    deeplab_model = DeepLabV3(num_classes=21)
    
    print("Модели созданы успешно!")
    print(f"YOLO параметры: {sum(p.numel() for p in yolo_model.parameters()):,}")
    print(f"DeepLab параметры: {sum(p.numel() for p in deeplab_model.parameters()):,}")

if __name__ == "__main__":
    main()
