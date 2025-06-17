import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import random
from sklearn.metrics import roc_auc_score

class SiameseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.person_groups = df.groupby('person_id')['image_path'].apply(list).to_dict()
        self.person_ids = list(self.person_groups.keys())
        
    def __len__(self):
        return len(self.df) * 2  # Увеличиваем размер для баланса positive/negative пар
    
    def __getitem__(self, idx):
        # Случайно выбираем, создать положительную или отрицательную пару
        is_same = random.random() > 0.5
        
        if is_same:
            # Положительная пара - два изображения одного человека
            person_id = random.choice(self.person_ids)
            if len(self.person_groups[person_id]) < 2:
                # Если у человека только одно фото, делаем отрицательную пару
                is_same = False
            else:
                img1_path, img2_path = random.sample(self.person_groups[person_id], 2)
        
        if not is_same:
            # Отрицательная пара - изображения разных людей
            person1_id, person2_id = random.sample(self.person_ids, 2)
            img1_path = random.choice(self.person_groups[person1_id])
            img2_path = random.choice(self.person_groups[person2_id])
        
        # Загружаем изображения
        img1 = Image.open(f"{self.img_dir}/{img1_path}").convert('RGB')
        img2 = Image.open(f"{self.img_dir}/{img2_path}").convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(1.0 if is_same else 0.0)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1)
        
    def forward_one(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        # L1 distance
        diff = torch.abs(out1 - out2)
        diff = self.dropout(diff)
        diff = F.relu(self.fc1(diff))
        diff = self.dropout(diff)
        out = torch.sigmoid(self.fc2(diff))
        
        return out

def train_siamese(model, train_loader, val_loader, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2).squeeze()
                val_loss += criterion(outputs, labels).item()
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_outputs)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {auc:.4f}')

def main():
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Данные
    df = pd.read_csv('train.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['person_id'])
    
    train_dataset = SiameseDataset(train_df, 'train_images/', transform)
    val_dataset = SiameseDataset(val_df, 'train_images/', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Модель и обучение
    model = SiameseNetwork()
    train_siamese(model, train_loader, val_loader)
    
    torch.save(model.state_dict(), 'siamese_model.pth')

if __name__ == "__main__":
    main()