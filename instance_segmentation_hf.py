"""
Сценарий 3: Инстанс-сегментация с использованием предобученных моделей HuggingFace
Этот код подходит для задач где нужно выделить отдельные экземпляры объектов
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import (
    AutoImageProcessor, 
    MaskFormerForInstanceSegmentation,
    Mask2FormerForUniversalSegmentation,
    AutoModelForUniversalSegmentation
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import json
from sklearn.metrics import average_precision_score
import cv2
from pycocotools import mask as coco_mask
from typing import List, Dict, Tuple

class COCOInstanceDataset(Dataset):
    """
    Датасет для инстанс-сегментации в формате COCO
    Ожидает аннотации в формате COCO JSON
    """
    def __init__(self, images_dir, annotations_file, processor=None, transform=None):
        self.images_dir = images_dir
        self.processor = processor
        self.transform = transform
        
        # Загружаем аннотации COCO
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Создаем словари для быстрого доступа
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Группируем аннотации по изображениям
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Загружаем изображение
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Получаем аннотации для этого изображения
        annotations = self.image_annotations.get(img_id, [])
        
        # Подготавливаем target для обучения
        target = self._prepare_target(annotations, img_info)
        
        # Применяем обработку
        if self.processor:
            # HuggingFace processor
            inputs = self.processor(image, target, return_tensors="pt")
            return inputs
        else:
            # Стандартные трансформации
            if self.transform:
                image = self.transform(image)
            return image, target
    
    def _prepare_target(self, annotations, img_info):
        """Подготавливает target в нужном формате"""
        target = {
            'image_id': img_info['id'],
            'annotations': annotations,
            'orig_size': (img_info['height'], img_info['width'])
        }
        return target

class HuggingFaceInstanceSegmentation:
    """
    Класс для работы с предобученными моделями инстанс-сегментации из HuggingFace
    """
    def __init__(self, model_name="facebook/mask2former-swin-base-coco-instance", device='cuda'):
        self.device = device
        self.model_name = model_name
        
        # Загружаем процессор и модель
        print(f"Loading {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        if "mask2former" in model_name.lower():
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        elif "maskformer" in model_name.lower():
            self.model = MaskFormerForInstanceSegmentation.from_pretrained(model_name)
        else:
            self.model = AutoModelForUniversalSegmentation.from_pretrained(model_name)
        
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        Выполняет предсказание для одного изображения
        """
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        
        # Подготавливаем входные данные
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Выполняем предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Постобработка результатов
        results = self.processor.post_process_instance_segmentation(
            outputs, 
            threshold=threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
        
        return {
            'image': image,
            'masks': results['segmentation'],
            'labels': results['segments_info'],
            'original_size': image.size
        }
    
    def predict_batch(self, image_paths: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Выполняет предсказание для батча изображений
        """
        results = []
        
        for img_path in image_paths:
            try:
                result = self.predict(img_path, threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append(None)
        
        return results
    
    def fine_tune(self, train_dataset, val_dataset, num_epochs=10, learning_rate=1e-5):
        """
        Дообучение модели на новых данных
        """
        # Переводим модель в режим обучения
        self.model.train()
        
        # Настраиваем оптимизатор
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=self._collate_fn)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Перемещаем данные на устройство
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    val_loss += outputs.loss.item()
            
            # Обновляем метрики
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Возвращаем модель в режим оценки
        self.model.eval()
        
        return train_losses, val_losses
    
    def _collate_fn(self, batch):
        """Функция для объединения батча"""
        return batch[0]  # Упрощенная версия

def visualize_instance_segmentation(results: Dict, save_path: str = None):
    """
    Визуализирует результаты инстанс-сегментации
    """
    image = results['image']
    masks = results['masks']
    labels = results['labels']
    
    # Создаем фигуру
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Маски
    if masks is not None and len(masks) > 0:
        # Объединяем все маски в одну для визуализации
        combined_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
        colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
        
        for i, (mask, label_info) in enumerate(zip(masks, labels)):
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            combined_mask[mask_np > 0] = (i + 1) * 30  # Разные значения для разных инстансов
        
        axes[1].imshow(combined_mask, cmap='tab20')
        axes[1].set_title(f'Instance Masks ({len(masks)} instances)')
        axes[1].axis('off')
        
        # Наложение масок на оригинальное изображение
        image_np = np.array(image)
        overlay = image_np.copy()
        
        for i, (mask, label_info) in enumerate(zip(masks, labels)):
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            if mask_np.sum() > 0:  # Если маска не пустая
                color = (colors[i % len(colors)][:3] * 255).astype(np.uint8)
                overlay[mask_np > 0] = overlay[mask_np > 0] * 0.7 + color * 0.3
        
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No instances detected', ha='center', va='center')
        axes[1].axis('off')
        axes[2].imshow(image)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_map_score(predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Вычисляет mAP (mean Average Precision) для инстанс-сегментации
    Упрощенная версия для демонстрации
    """
    # Это упрощенная реализация
    # В реальной задаче используйте официальные метрики COCO
    
    all_scores = []
    all_labels = []
    
    for pred, gt in zip(predictions, ground_truth):
        if pred is None:
            continue
            
        pred_masks = pred.get('masks', [])
        pred_labels = pred.get('labels', [])
        
        # Простая эвристика для демонстрации
        if len(pred_masks) > 0:
            scores = [0.8] * len(pred_masks)  # Заглушка для confidence scores
            labels = [1] * len(pred_masks)    # Заглушка для ground truth labels
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    if len(all_scores) == 0:
        return 0.0
    
    # Используем sklearn для вычисления AP
    return average_precision_score(all_labels, all_scores)

def create_submission_from_predictions(predictions: List[Dict], output_dir: str = 'submission'):
    """
    Создает файлы для сабмишена на основе предсказаний
    """
    os.makedirs(output_dir, exist_ok=True)
    
    submission_data = []
    
    for i, pred in enumerate(predictions):
        if pred is None:
            continue
        
        masks = pred.get('masks', [])
        labels = pred.get('labels', [])
        
        for j, (mask, label_info) in enumerate(zip(masks, labels)):
            # Конвертируем маску в RLE формат (упрощенная версия)
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            
            # Сохраняем маску как изображение
            mask_filename = f'mask_{i}_{j}.png'
            mask_path = os.path.join(output_dir, mask_filename)
            
            # Конвертируем в бинарную маску
            binary_mask = (mask_np > 0).astype(np.uint8) * 255
            cv2.imwrite(mask_path, binary_mask)
            
            # Добавляем в submission
            submission_data.append({
                'image_id': i,
                'instance_id': j,
                'mask_path': mask_filename,
                'confidence': label_info.get('score', 0.5) if isinstance(label_info, dict) else 0.5,
                'category_id': label_info.get('id', 1) if isinstance(label_info, dict) else 1
            })
    
    # Сохраняем CSV файл
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
    
    print(f"Created submission with {len(submission_data)} instances in {output_dir}")

# Пример использования
if __name__ == "__main__":
    # Параметры
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Список доступных предобученных моделей
    available_models = [
        "facebook/mask2former-swin-base-coco-instance",
        "facebook/mask2former-swin-large-coco-instance", 
        "facebook/maskformer-swin-base-coco",
        "facebook/maskformer-swin-large-coco"
    ]
    
    print("Available models:")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")
    
    # Выбираем модель (можно изменить на любую из списка)
    selected_model = available_models[0]
    print(f"\nUsing model: {selected_model}")
    
    try:
        # Создаем объект для инстанс-сегментации
        segmentator = HuggingFaceInstanceSegmentation(
            model_name=selected_model, 
            device=DEVICE
        )
        
        print("Model loaded successfully!")
        
        # Пример предсказания для одного изображения
        # (раскомментировать при наличии изображения)
        # results = segmentator.predict('path/to/your/image.jpg', threshold=0.5)
        # visualize_instance_segmentation(results, save_path='prediction_result.png')
        
        # Пример батчевого предсказания
        # image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
        # batch_results = segmentator.predict_batch(image_paths, threshold=0.5)
        
        # Создание сабмишена
        # create_submission_from_predictions(batch_results, 'submission')
        
        # Пример дообучения (раскомментировать при наличии данных)
        # train_dataset = COCOInstanceDataset('train_images/', 'train_annotations.json', 
        #                                   processor=segmentator.processor)
        # val_dataset = COCOInstanceDataset('val_images/', 'val_annotations.json', 
        #                                 processor=segmentator.processor)
        
        # train_losses, val_losses = segmentator.fine_tune(train_dataset, val_dataset, 
        #                                                 num_epochs=5, learning_rate=1e-5)
        
        # Визуализация процесса обучения
        # plt.figure(figsize=(10, 5))
        # plt.plot(train_losses, label='Train Loss')
        # plt.plot(val_losses, label='Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.title('Fine-tuning Progress')
        # plt.show()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet connection for the first model download")
        print("Or use a different model from the available list")
    
    print("\nInstance segmentation setup complete!")
    print("To use this code:")
    print("1. Prepare your dataset in COCO format")
    print("2. Update image and annotation paths")
    print("3. Run prediction or fine-tuning")
    print("4. Create submission files")
