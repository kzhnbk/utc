import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math

# Простой токенизатор
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        self.vocab = {word: i+2 for i, word in enumerate(sorted(words))}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text, max_len=128):
        tokens = [self.vocab.get(word.lower(), 1) for word in text.split()]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        return tokens
    
    def decode(self, tokens):
        return ' '.join([self.idx2word.get(t, '<unk>') for t in tokens if t != 0])

# Dataset класс
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = torch.tensor(self.src_tokenizer.encode(self.src_texts[idx], self.max_len))
        tgt = torch.tensor(self.tgt_tokenizer.encode(self.tgt_texts[idx], self.max_len))
        return src, tgt

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Простой Transformer
class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, 
                 num_layers=6, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.output_proj(output)

# Основная функция обучения
def train_translation_model():
    # Пример данных (замените на реальные)
    src_texts = ["hello world", "how are you", "good morning", "see you later"] * 100
    tgt_texts = ["привет мир", "как дела", "доброе утро", "увидимся позже"] * 100
    
    # Создание токенизаторов
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.build_vocab(src_texts)
    tgt_tokenizer.build_vocab(tgt_texts)
    
    # Создание датасета
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Модель
    model = SimpleTransformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Обучение
    model.train()
    for epoch in range(10):
        total_loss = 0
        for src, tgt in dataloader:
            optimizer.zero_grad()
            
            # Сдвиг для decoder input/target
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]
            
            output = model(src, decoder_input)
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, src_tokenizer, tgt_tokenizer

if __name__ == "__main__":
    model, src_tok, tgt_tok = train_translation_model()
    print("Модель обучена!")