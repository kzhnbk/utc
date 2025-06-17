import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModel
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import pickle

class FaceEmbeddingDataset(Dataset):
    def __init__(self, df, img_dir, feature_extractor):
        self.df = df
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        
        image = Image.open(img_path).convert('RGB')
        
        # Используем feature extractor от HuggingFace
        inputs = self.feature_extractor(image, return_tensors="pt")
        
        if 'person_id' in row:
            return inputs['pixel_values'].squeeze(0), row['person_id']
        else:
            return inputs['pixel_values'].squeeze(0), row.get('image_id', idx)

class FaceRecognitionHF:
    def __init__(self, model_name="microsoft/DiT-base-distilled-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем предобученную модель
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.embeddings = {}
        self.person_embeddings = {}
        
    def extract_embeddings(self, dataloader):
        """Извлечение эмбеддингов из изображений"""
        
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels) in enumerate(dataloader):
                images = images.to(self.device)
                
                # Получаем эмбеддинги
                outputs = self.model(images)
                
                # Используем pooled output или last hidden state
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.extend(batch_labels.numpy() if isinstance(batch_labels, torch.Tensor) else batch_labels)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * len(images)} images")
        
        embeddings = np.vstack(embeddings)
        return embeddings, labels
    
    def build_person_database(self, train_df, img_dir):
        """Создание базы эмбеддингов для известных людей"""
        
        train_dataset = FaceEmbeddingDataset(train_df, img_dir, self.feature_extractor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        
        embeddings, person_ids = self.extract_embeddings(train_loader)
        
        # Группируем эмбеддинги по person_id
        for emb, pid in zip(embeddings, person_ids):
            if pid not in self.person_embeddings:
                self.person_embeddings[pid] = []
            self.person_embeddings[pid].append(emb)
        
        # Вычисляем средние эмбеддинги для каждого человека
        self.avg_person_embeddings = {}
        for pid, embs in self.person_embeddings.items():
            self.avg_person_embeddings[pid] = np.mean(embs, axis=0)
        
        print(f"Built database for {len(self.avg_person_embeddings)} persons")
        
    def recognize_faces(self, test_df, img_dir, threshold=0.5):
        """Распознавание лиц на тестовых изображениях"""
        
        test_dataset = FaceEmbeddingDataset(test_df, img_dir, self.feature_extractor)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        test_embeddings, image_ids = self.extract_embeddings(test_loader)
        
        predictions = []
        confidences = []
        
        for emb in test_embeddings:
            best_match = None
            best_similarity = -1
            
            # Сравниваем с каждым человеком в базе
            for pid, avg_emb in self.avg_person_embeddings.items():
                similarity = cosine_similarity([emb], [avg_emb])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pid
            
            # Принимаем решение на основе порога
            if best_similarity > threshold:
                predictions.append(best_match)
                confidences.append(best_similarity)
            else:
                predictions.append(-1)  # Неизвестный человек
                confidences.append(best_similarity)
        
        return predictions, confidences, image_ids
    
    def cluster_unknown_faces(self, embeddings, eps=0.5, min_samples=2):
        """Кластеризация неизвестных лиц"""
        
        # Используем DBSCAN для кластеризации
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        return cluster_labels

def main():
    # Инициализация модели
    face_recognizer = FaceRecognitionHF("google/vit-base-patch16-224")
    
    # Загрузка данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("Building person database...")
    face_recognizer.build_person_database(train_df, 'train_images/')
    
    print("Recognizing faces in test set...")
    predictions, confidences, image_ids = face_recognizer.recognize_faces(
        test_df, 'test_images/', threshold=0.6
    )
    
    # Создание submission
    submission = pd.DataFrame({
        'image_id': image_ids,
        'person_id': predictions,
        'confidence': confidences
    })
    
    # Обработка неизвестных лиц
    unknown_mask = np.array(predictions) == -1
    if np.sum(unknown_mask) > 0:
        print(f"Found {np.sum(unknown_mask)} unknown faces, clustering...")
        
        # Получаем эмбеддинги неизвестных лиц
        test_dataset = FaceEmbeddingDataset(test_df, 'test_images/', face_recognizer.feature_extractor)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        all_embeddings, _ = face_recognizer.extract_embeddings(test_loader)
        
        unknown_embeddings = all_embeddings[unknown_mask]
        cluster_labels = face_recognizer.cluster_unknown_faces(unknown_embeddings)
        
        # Присваиваем новые ID для кластеров
        max_person_id = max(face_recognizer.avg_person_embeddings.keys())
        cluster_to_id = {}
        
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id != -1:  # -1 это шум в DBSCAN
                if cluster_id not in cluster_to_id:
                    max_person_id += 1
                    cluster_to_id[cluster_id] = max_person_id
                
                # Обновляем предсказание
                original_idx = np.where(unknown_mask)[0][i]
                predictions[original_idx] = cluster_to_id[cluster_id]
    
    # Финальный submission
    final_submission = pd.DataFrame({
        'image_id': image_ids,
        'person_id': predictions
    })
    
    final_submission.to_csv('face_recognition_submission.csv', index=False)
    
    # Сохранение модели и эмбеддингов
    with open('person_embeddings.pkl', 'wb') as f:
        pickle.dump(face_recognizer.avg_person_embeddings, f)
    
    print(f"Recognition complete!")
    print(f"Recognized persons: {len(set(p for p in predictions if p != -1))}")
    print(f"Unknown faces: {sum(1 for p in predictions if p == -1)}")

# Дополнительная функция для fine-tuning модели
def finetune_model(train_df, img_dir, num_epochs=5):
    """Fine-tuning предобученной модели для лучшего качества"""
    
    from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
    from torch.utils.data import Dataset
    
    class FinetuneDataset(Dataset):
        def __init__(self, df, img_dir, processor):
            self.df = df
            self.img_dir = img_dir
            self.processor = processor
            
            # Преобразуем person_id в числовые метки
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(df['person_id'])
            self.num_classes = len(self.label_encoder.classes_)
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['image_path'])
            
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    # Загрузка модели для классификации
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=train_df['person_id'].nunique(),
        ignore_mismatched_sizes=True
    )
    
    # Датасет
    dataset = FinetuneDataset(train_df, img_dir, processor)
    
    # Настройки обучения
    training_args = TrainingArguments(
        output_dir='./face_model_finetuned',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        remove_unused_columns=False,
    )
    
    # Тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
    )
    
    # Обучение
    trainer.train()
    
    # Сохранение
    trainer.save_model()
    processor.save_pretrained('./face_model_finetuned')
    
    return model, processor, dataset.label_encoder

if __name__ == "__main__":
    main()