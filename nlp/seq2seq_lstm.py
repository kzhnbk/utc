import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import random

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_vocab_size)
        
        # Attention (простая версия)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.output_proj.out_features
        
        # Encoder
        src_emb = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(src_emb)
        
        # Decoder
        decoder_outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        decoder_input = tgt[:, 0].unsqueeze(1)  # Start token
        
        for t in range(1, tgt_len):
            decoder_emb = self.decoder_embedding(decoder_input)
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_emb, (hidden, cell))
            
            # Simple attention
            attn_weights = self.compute_attention(decoder_output, encoder_outputs)
            context = torch.bmm(attn_weights, encoder_outputs)
            
            # Combine context and decoder output
            combined = torch.cat([decoder_output, context], dim=2)
            output = self.output_proj(combined)
            
            decoder_outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=2)
        
        return decoder_outputs
    
    def compute_attention(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attention(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1).unsqueeze(1)

def train_seq2seq():
    # Пример данных
    src_texts = ["hello", "good morning", "how are you", "thank you"] * 100
    tgt_texts = ["привет", "доброе утро", "как дела", "спасибо"] * 100
    
    # Создание словарей
    src_vocab = {word: i+2 for i, word in enumerate(set(' '.join(src_texts).split()))}
    src_vocab.update({'<pad>': 0, '<unk>': 1})
    
    tgt_vocab = {word: i+2 for i, word in enumerate(set(' '.join(tgt_texts).split()))}
    tgt_vocab.update({'<pad>': 0, '<unk>': 1})
    
    def text_to_ids(text, vocab, max_len=20):
        ids = [vocab.get(word, 1) for word in text.split()]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        return ids
    
    # Подготовка данных
    src_ids = [text_to_ids(text, src_vocab) for text in src_texts]
    tgt_ids = [text_to_ids(text, tgt_vocab) for text in tgt_texts]
    
    src_tensor = torch.tensor(src_ids)
    tgt_tensor = torch.tensor(tgt_ids)
    
    dataset = torch.utils.data.TensorDataset(src_tensor, tgt_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Модель
    model = Seq2SeqLSTM(len(src_vocab), len(tgt_vocab))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    model.train()
    for epoch in range(15):
        total_loss = 0
        for src_batch, tgt_batch in dataloader:
            optimizer.zero_grad()
            
            output = model(src_batch, tgt_batch)
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), 
                           tgt_batch[:, 1:].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, src_vocab, tgt_vocab

if __name__ == "__main__":
    model, src_vocab, tgt_vocab = train_seq2seq()
    print("Seq2Seq модель обучена!")