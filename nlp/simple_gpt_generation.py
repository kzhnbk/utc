import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))
        
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, 
                 max_seq_len=512, d_ff=1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Создание causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(x.device)
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        x = token_emb + pos_emb
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits

def train_simple_gpt():
    # Пример текстов
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is changing the world",
        "machine learning models can learn from data",
        "deep learning uses neural networks",
        "python is a programming language"
    ] * 100
    
    # Создание словаря
    all_words = set()
    for text in texts:
        all_words.update(text.split())
    
    vocab = {word: i+1 for i, word in enumerate(sorted(all_words))}
    vocab['<pad>'] = 0
    idx_to_word = {v: k for k, v in vocab.items()}
    
    def text_to_ids(text, max_len=64):
        ids = [vocab.get(word, 0) for word in text.split()]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids
    
    # Подготовка данных
    sequences = [text_to_ids(text) for text in texts]
    sequences = torch.tensor(sequences)
    
    # Модель
    model = SimpleGPT(vocab_size=len(vocab), d_model=128, num_heads=4, num_layers=4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Обучение
    model.train()
    batch_size = 16
    
    for epoch in range(20):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Input и target
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/num_batches:.4f}')
    
    return model, vocab, idx_to_word

def generate_text_gpt(model, vocab, idx_to_word, prompt, max_length=50, temperature=0.8):
    model.eval()
    
    # Токенизация промпта
    prompt_ids = [vocab.get(word, 0) for word in prompt.split()]
    input_ids = torch.tensor([prompt_ids])
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Остановка на padding token
            if next_token.item() == 0:
                break
    
    # Декодирование
    generated_ids = generated[0].tolist()
    generated_text = ' '.join([idx_to_word.get(id, '<unk>') for id in generated_ids if id != 0])
    
    return generated_text

if __name__ == "__main__":
    model, vocab, idx_to_word = train_simple_gpt()
    
    # Тест генерации
    prompt = "artificial intelligence"
    generated = generate_text_gpt(model, vocab, idx_to_word, prompt)
    print(f"Generated: {generated}")