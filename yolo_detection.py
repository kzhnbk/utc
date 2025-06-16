"""
Сценарий 1: Обнаружение объектов с использованием YOLO архитектуры
Этот код подходит для задач где нужно найти и классифицировать объекты на изображениях
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torchvision.ops import nms

class YOLODataset(Dataset):
    """
    Датасет для YOLO формата данных
    Ожидает CSV файл с колонками: image_path, x1, y1, x2, y2, class_id
    где (x1,y1) и (x2,y2) - координаты bounding box в нормализованном виде [0,1]
    """
    def __init__(self, csv_path, img_dir, img_size=416, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        
        # Группируем аннотации по изображениям
        self.image_groups = self.data.groupby('image_path')
        self.image_paths = list(self.image_groups.groups.keys())
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Загружаем изображение
        full_img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(full_img_path).convert('RGB')
        
        # Получаем все bounding boxes для этого изображения
        img_annotations = self.image_groups.get_group(img_path)
        
        # Конвертируем в тензоры
        boxes = []
        labels = []
        
        for _, row in img_annotations.iterrows():
            # Координаты уже должны быть нормализованы к [0,1]
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            boxes.append([center_x, center_y, width, height])
            labels.append(row['class_id'])
        
        if self.transform:
            image = self.transform(image)
        
        # Создаем target тензор для YOLO: [class_id, center_x, center_y, width, height]
        target = torch.zeros((len(boxes), 5))
        for i, (box, label) in enumerate(zip(boxes, labels)):
            target[i] = torch.tensor([label] + box)
            
        return image, target

class YOLOBlock(nn.Module):
    """Базовый блок для YOLO архитектуры"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class SimpleYOLO(nn.Module):
    """
    Упрощенная версия YOLO для демонстрации концепции
    В реальной олимпиаде лучше использовать предобученные модели
    """
    def __init__(self, num_classes=20, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone - упрощенная версия
        self.backbone = nn.Sequential(
            YOLOBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            YOLOBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            YOLOBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            YOLOBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            YOLOBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            YOLOBlock(512, 1024, 3, 1, 1),
        )
        
        # Detection head
        # Выход: [batch, anchors * (5 + num_classes), height, width]
        # 5 = x, y, w, h, confidence
        self.detection_head = nn.Conv2d(1024, num_anchors * (5 + num_classes), 1)
        
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections

def yolo_loss(predictions, targets, num_classes=20, num_anchors=3):
    """
    Упрощенная версия YOLO loss функции
    В реальной реализации нужно учитывать anchor boxes и IoU
    """
    batch_size, _, grid_h, grid_w = predictions.shape
    
    # Reshape predictions: [batch, anchors, grid_h, grid_w, 5+num_classes]
    predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
    
    # Простая MSE loss для демонстрации
    # В реальной реализации используется более сложная loss с IoU
    total_loss = 0
    
    for i in range(batch_size):
        if len(targets[i]) > 0:
            # Здесь должна быть логика сопоставления предсказаний с ground truth
            # Для простоты используем MSE loss
            pred_conf = torch.sigmoid(predictions[i, :, :, :, 4])
            total_loss += torch.mean(pred_conf ** 2)  # Simplified objectness loss
    
    return total_loss / batch_size

def train_yolo_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Функция обучения YOLO модели"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = yolo_loss(predictions, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                predictions = model(images)
                loss = yolo_loss(predictions, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def predict_and_visualize(model, image_path, device='cuda', conf_threshold=0.5):
    """Функция для предсказания и визуализации результатов"""
    model.eval()
    
    # Загружаем и предобрабатываем изображение
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
        
        # Здесь должна быть логика декодирования предсказаний
        # Для простоты выводим размер выходного тензора
        print(f"Prediction shape: {predictions.shape}")
    
    # Визуализация (упрощенная версия)
    plt.figure(figsize=(10, 8))
    plt.imshow(original_image)
    plt.title('Object Detection Results')
    plt.axis('off')
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Параметры
    IMG_SIZE = 416
    BATCH_SIZE = 8
    NUM_CLASSES = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Трансформации для обучения
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создание датасетов (пути нужно изменить на актуальные)
    # train_dataset = YOLODataset('train_annotations.csv', 'train_images/', transform=train_transform)
    # val_dataset = YOLODataset('val_annotations.csv', 'val_images/', transform=val_transform)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Создание модели
    model = SimpleYOLO(num_classes=NUM_CLASSES)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Обучение модели (раскомментировать при наличии данных)
    # train_losses, val_losses = train_yolo_model(model, train_loader, val_loader, num_epochs=50, device=DEVICE)
    
    # Визуализация лоссов
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training Progress')
    # plt.show()
    
    print("YOLO model setup complete. Ready for training with your dataset!")
