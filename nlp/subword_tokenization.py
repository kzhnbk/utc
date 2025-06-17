import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class BPETokenizer:
    """Byte Pair Encoding токенизатор с нуля"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.word_freq = {}
        
    def get_word_freq(self, texts):
        """Подсчет частоты слов"""
        word_freq = defaultdict(int)
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                word_freq[word] += 1
        return dict(word_freq)
    
    def get_pairs(self, word_freq):
        """Получение пар символов/подслов"""
        pairs = defaultdict(int)
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, word_freq):
        """Слияние наиболее частой пары"""
        new_word_freq = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freq:
            new_word = p.sub(''.join(pair), word)
            new_word_freq[new_word] = word_freq[word]
        return new_word_freq
    
    def train(self, texts):
        """Обучение BPE"""
        # Инициализация словаря символами
        word_freq = self.get_word_freq(texts)
        
        # Разбиваем слова на символы
        for word in word_freq:
            self.word_freq[' '.join(list(word))] = word_freq[word]
        
        # Строим словарь из символов
        for word in self.word_freq:
            for char in word.split():
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        # BPE алгоритм
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self.get_pairs(self.word_freq)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            self.word_freq = self.merge_vocab(best_pair, self.word_freq)
            self.merges.append(best_pair)
            
            # Добавляем новый токен в словарь
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
    
    def tokenize(self, text):
        """Токенизация текста"""
        words = re.findall(r'\b\w+\b', text.lower())
        tokens = []
        
        for word in words:
            # Разбиваем на символы
            word_tokens = list(word)
            
            # Применяем слияния
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = (word_tokens[:i] + 
                                     [''.join(pair)] + 
                                     word_tokens[i + 2:])
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text):
        """Кодирование в индексы"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, 0) for token in tokens]  # 0 для UNK

class SubwordLM(nn.Module):
    """Простая языковая модель на subword токенах"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.output(self.dropout(lstm_out))
        return output

def train_bpe_language_model():
    # Пример корпуса
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming technology",
        "machine learning algorithms process data efficiently",
        "natural language processing enables text analysis",
        "deep learning models require large datasets",
        "neural networks learn complex patterns automatically",
        "tokenization breaks text into meaningful units",
        "lemmatization reduces words to base forms"
    ]
    
    print("Обучение BPE токенизатора...")
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(corpus)
    
    print(f"Размер словаря: {len(tokenizer.vocab)}")
    print(f"Количество слияний: {len(tokenizer.merges)}")
    
    # Тестирование токенизации
    test_text = "transforming algorithms automatically"
    tokens = tokenizer.tokenize(test_text)
    encoded = tokenizer.encode(test_text)
    
    print(f"\nТест токенизации:")
    print(f"Текст: {test_text}")
    print(f"Токены: {tokens}")
    print(f"Индексы: {encoded}")
    
    # Подготовка данных для языковой модели
    sequences = []
    for text in corpus:
        encoded = tokenizer.encode(text)
        # Создаем последовательности для обучения
        for i in range(len(encoded) - 1):
            sequences.append((encoded[i:i+5], encoded[i+1:i+6]))  # окно 5
    
    if sequences:
        # Создание модели
        model = SubwordLM(len(tokenizer.vocab))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Простое обучение
        model.train()
        for epoch in range(10):
            total_loss = 0
            for x_seq, y_seq in sequences[:5]:  # Первые 5 для примера
                if len(x_seq) == len(y_seq) == 5:
                    x_tensor = torch.LongTensor(x_seq).unsqueeze(0)
                    y_tensor = torch.LongTensor(y_seq).unsqueeze(0)
                    
                    optimizer.zero_grad()
                    output = model(x_tensor)
                    loss = criterion(output.view(-1, len(tokenizer.vocab)), y_tensor.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/5:.4f}")
    
    return tokenizer, model if 'model' in locals() else None

def compare_tokenization_methods():
    """Сравнение разных методов токенизации"""
    text = "unhappiness preprocessing tokenization"
    
    # Обычная токенизация
    word_tokens = text.split()
    
    # Символьная токенизация
    char_tokens = list(text.replace(' ', '_'))
    
    # BPE токенизация
    bpe_tokenizer = BPETokenizer(vocab_size=100)
    bpe_tokenizer.train([text])
    bpe_tokens = bpe_tokenizer.tokenize(text)
    
    print("Сравнение методов токенизации:")
    print(f"Исходный текст: {text}")
    print(f"Слова ({len(word_tokens)}): {word_tokens}")
    print(f"Символы ({len(char_tokens)}): {char_tokens}")
    print(f"BPE ({len(bpe_tokens)}): {bpe_tokens}")
    
    return {
        'word': word_tokens,
        'char': char_tokens,
        'bpe': bpe_tokens
    }

if __name__ == "__main__":
    # Обучение BPE модели
    tokenizer, model = train_bpe_language_model()
    
    # Сравнение методов
    comparison = compare_tokenization_methods()