import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

class AnomalyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data)
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_autoencoder(model, train_loader, val_loader, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            if isinstance(model, VAE):
                recon, mu, logvar = model(data)
                loss = vae_loss(recon, data, mu, logvar)
            else:
                recon = model(data)
                loss = criterion(recon, data)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                if isinstance(model, VAE):
                    recon, mu, logvar = model(data)
                    loss = vae_loss(recon, data, mu, logvar)
                else:
                    recon = model(data)
                    loss = criterion(recon, data)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def detect_anomalies(model, data_loader, threshold_percentile=95):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            if isinstance(model, VAE):
                recon, _, _ = model(data)
            else:
                recon = model(data)
            
            # Вычисляем ошибку реконструкции
            mse = torch.mean((data - recon) ** 2, dim=1)
            reconstruction_errors.extend(mse.cpu().numpy())
    
    # Определяем порог
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Аномалии - это точки с ошибкой выше порога
    anomalies = np.array(reconstruction_errors) > threshold
    
    return reconstruction_errors, anomalies, threshold

def main():
    # Загрузка данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Предполагаем, что в train есть колонка 'is_anomaly' для валидации
    features = [col for col in train_df.columns if col not in ['id', 'is_anomaly']]
    
    X_train = train_df[features].values
    X_test = test_df[features].values
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Для обучения используем только нормальные данные
    if 'is_anomaly' in train_df.columns:
        normal_mask = train_df['is_anomaly'] == 0
        X_train_normal = X_train_scaled[normal_mask]
    else:
        X_train_normal = X_train_scaled
    
    # Разделение на train/val
    split_idx = int(0.8 * len(X_train_normal))
    X_train_split = X_train_normal[:split_idx]
    X_val_split = X_train_normal[split_idx:]
    
    # Датасеты
    train_dataset = AnomalyDataset(X_train_split)
    val_dataset = AnomalyDataset(X_val_split)
    test_dataset = AnomalyDataset(X_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_dim = X_train_scaled.shape[1]
    
    # 1. Обычный Autoencoder
    print("Training Autoencoder...")
    autoencoder = Autoencoder(input_dim, encoding_dim=input_dim//4)
    train_autoencoder(autoencoder, train_loader, val_loader, epochs=100)
    
    # 2. VAE
    print("Training VAE...")
    vae = VAE(input_dim, latent_dim=input_dim//8)
    train_autoencoder(vae, train_loader, val_loader, epochs=100)
    
    # Детекция аномалий
    print("Detecting anomalies...")
    
    # Autoencoder
    ae_errors, ae_anomalies, ae_threshold = detect_anomalies(autoencoder, test_loader)
    
    # VAE
    vae_errors, vae_anomalies, vae_threshold = detect_anomalies(vae, test_loader)
    
    # Ансамбль - берем среднее предсказаний
    ensemble_anomalies = (ae_anomalies.astype(int) + vae_anomalies.astype(int)) >= 1
    
    # Создание submission
    submission = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'is_anomaly': ensemble_anomalies.astype(int)
    })
    
    submission.to_csv('anomaly_submission.csv', index=False)
    
    # Сохранение моделей
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    torch.save(vae.state_dict(), 'vae.pth')
    
    print(f"Autoencoder threshold: {ae_threshold:.6f}")
    print(f"VAE threshold: {vae_threshold:.6f}")
    print(f"Detected {np.sum(ensemble_anomalies)} anomalies out of {len(ensemble_anomalies)} samples")

if __name__ == "__main__":
    main()