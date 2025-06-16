"""
Сценарий 2: Семантическая сегментация с использованием U-Net архитектуры
Этот код подходит для задач попикельной классификации изображений
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import jaccard_score
import cv2

class SegmentationDataset(Dataset):
    """
    Датасет для семантической сегментации
    Ожидает CSV файл с колонками: image_path, mask_path
    где mask_path - путь к маске сегментации
    """
    def __init__(self, csv_path, img_dir, mask_dir, img_size=256, transform=None, mask_transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Загружаем изображение
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # Загружаем маску
        mask_path = os.path.join(self.mask_dir, row['mask_path'])
        mask = Image.open(mask_path).convert('L')  # Grayscale для масок
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Стандартная обработка маски
            mask = transforms.Resize((self.img_size, self.img_size))(mask)
            mask = transforms.ToTensor()(mask)
            mask = mask.squeeze(0).long()  # Убираем канальную размерность и конвертируем в long
            
        return image, mask

class DoubleConv(nn.Module):
    """Двойная свертка - базовый блок U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Понижающий путь (encoder) в U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Повышающий путь (decoder) в U-Net"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Компенсируем разность в размерах между x1 и x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Конкатенируем по каналам
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Финальный слой для вывода"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net архитектура для семантической сегментации
    Классическая реализация с skip connections
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (понижающий путь)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (повышающий путь)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder с skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DiceLoss(nn.Module):
    """
    Dice Loss - популярная loss функция для сегментации
    Особенно эффективна при дисбалансе классов
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Применяем sigmoid для получения вероятностей
        predictions = torch.sigmoid(predictions)
        
        # Flatten тензоры
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Вычисляем пересечение и объединение
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()
        
        # Dice коэффициент
        dice = (2 * intersection + self.smooth) / (total + self.smooth)
        
        # Возвращаем loss (1 - dice)
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Комбинированная loss функция: BCE + Dice Loss
    Часто показывает лучшие результаты чем каждая loss по отдельности
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets.float())
        dice_loss = self.dice(predictions, targets.float())
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """Вычисляет IoU (Intersection over Union) метрику"""
    pred_mask = (pred_mask > threshold).astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    if union == 0:
        return 1.0  # Если оба маски пустые
    
    return intersection / union

def train_unet_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Функция обучения U-Net модели"""
    model = model.to(device)
    
    # Используем комбинированную loss функцию
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_ious = []
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Убеждаемся что размеры совпадают
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs.squeeze(1), masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        total_iou = 0
        num_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(outputs.squeeze(1), masks)
                val_loss += loss.item()
                
                # Вычисляем IoU для каждого изображения в батче
                pred_masks = torch.sigmoid(outputs).cpu().numpy()
                true_masks = masks.cpu().numpy()
                
                for pred, true in zip(pred_masks, true_masks):
                    iou = calculate_iou(pred.squeeze(), true)
                    total_iou += iou
                    num_samples += 1
        
        # Вычисляем средние метрики
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        avg_iou = total_iou / num_samples
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(avg_iou)
        
        scheduler.step(val_loss)
        
        # Сохраняем лучшую модель
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), 'best_unet_model.pth')
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {avg_iou:.4f}')
    
    return train_losses, val_losses, val_ious

def visualize_predictions(model, dataset, num_samples=4, device='cuda'):
    """Визуализация предсказаний модели"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
    
    for i in range(num_samples):
        image, true_mask = dataset[i]
        
        # Предсказание
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            pred_logits = model(image_batch)
            pred_mask = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]
        
        # Денормализация изображения для визуализации
        image_np = image.permute(1, 2, 0).numpy()
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        # Визуализация
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask.numpy(), cmap='gray')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask (IoU: {calculate_iou(pred_mask, true_mask.numpy()):.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_submission_masks(model, test_loader, device='cuda', output_dir='submission_masks'):
    """Создание масок для тестового набора данных"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    submission_data = []
    
    with torch.no_grad():
        for batch_idx, (images, image_names) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            pred_masks = torch.sigmoid(outputs).cpu().numpy()
            
            for i, (mask, img_name) in enumerate(zip(pred_masks, image_names)):
                # Конвертируем в бинарную маску
                binary_mask = (mask[0] > 0.5).astype(np.uint8) * 255
                
                # Сохраняем маску
                mask_filename = f"{img_name.split('.')[0]}_mask.png"
                mask_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, binary_mask)
                
                submission_data.append({
                    'image_name': img_name,
                    'mask_path': mask_filename
                })
    
    # Создаем submission CSV
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission.csv', index=False)
    print(f"Created {len(submission_data)} masks in {output_dir}")

# Пример использования
if __name__ == "__main__":
    # Параметры
    IMG_SIZE = 256
    BATCH_SIZE = 8
    NUM_CLASSES = 1  # Для бинарной сегментации
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # Создание датасетов (пути нужно изменить на актуальные)
    # train_dataset = SegmentationDataset('train.csv', 'train_images/', 'train_masks/', 
    #                                   transform=train_transform, mask_transform=mask_transform)
    # val_dataset = SegmentationDataset('val.csv', 'val_images/', 'val_masks/', 
    #                                 transform=val_transform, mask_transform=mask_transform)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Создание модели
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True)
    print(f"U-Net model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Обучение модели (раскомментировать при наличии данных)
    # train_losses, val_losses, val_ious = train_unet_model(model, train_loader, val_loader, 
    #                                                       num_epochs=50, device=DEVICE)
    
    # Визуализация прогресса обучения
    # plt.figure(figsize=(15, 5))
    # 
    # plt.subplot(1, 3, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training Progress - Loss')
    # 
    # plt.subplot(1, 3, 2)
    # plt.plot(val_ious, label='Validation IoU', color='green')
    # plt.xlabel('Epoch')
    # plt.ylabel('IoU')
    # plt.legend()
    # plt.title('Training Progress - IoU')
    # 
    # plt.subplot(1, 3, 3)
    # plt.plot(range(len(train_losses)), [1-x for x in train_losses], label='Train Accuracy Proxy')
    # plt.plot(range(len(val_losses)), [1-x for x in val_losses], label='Val Accuracy Proxy')
    # plt.xlabel('Epoch')
    # plt.ylabel('1 - Loss')
    # plt.legend()
    # plt.title('Accuracy Proxy')
    # 
    # plt.tight_layout()
    # plt.show()
    
    print("U-Net segmentation model setup complete. Ready for training with your dataset!")