import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class FaceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = row['person_id']
        return image, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_acc = correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')

# Основной код для обучения
def main():
    # Загрузка данных
    df = pd.read_csv('train.csv')  # колонки: image_path, person_id
    
    # Препроцессинг
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Разделение данных
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['person_id'])
    
    # Создание датасетов
    train_dataset = FaceDataset(train_df, 'train_images/', transform_train)
    val_dataset = FaceDataset(val_df, 'train_images/', transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Модель
    num_classes = df['person_id'].nunique()
    model = FaceRecognitionModel(num_classes)
    
    # Обучение
    train_model(model, train_loader, val_loader, epochs=20)
    
    # Сохранение модели
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    
    # Предсказание для тестовых данных
    test_df = pd.read_csv('test.csv')
    test_dataset = FaceDataset(test_df, 'test_images/', transform_val)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    # Создание submission файла
    submission = pd.DataFrame({
        'image_id': test_df['image_id'],
        'person_id': predictions
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()