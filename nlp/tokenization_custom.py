import pandas as pd
import numpy as np
import re
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CustomTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = None
        
    def build_vocab(self, texts):
        # Базовая токенизация
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Строим словарь
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        self.word2idx = {'<UNK>': 0, '<PAD>': 1}
        self.idx2word = {0: '<UNK>', 1: '<PAD>'}
        
        for i, (word, _) in enumerate(most_common):
            self.word2idx[word] = i + 2
            self.idx2word[i + 2] = word
            
        self.vocab = set(self.word2idx.keys())
        
    def tokenize(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.word2idx.get(word, 0) for word in words]
    
    def encode_batch(self, texts, max_len=100):
        encoded = []
        for text in texts:
            tokens = self.tokenize(text)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens.extend([1] * (max_len - len(tokens)))  # PAD
            encoded.append(tokens)
        return np.array(encoded)

# Простая модель классификации
class TextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = self.classifier(self.dropout(hidden[-1]))
        return output

# Пример использования
def train_tokenization_model():
    # Загрузка данных (замените на реальный датасет)
    data = pd.DataFrame({
        'text': ['This is positive text', 'This is negative text', 'Great movie!', 'Bad service'],
        'label': [1, 0, 1, 0]
    })
    
    # Токенизация
    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(data['text'].tolist())
    
    # Подготовка данных
    X = tokenizer.encode_batch(data['text'].tolist())
    y = data['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Модель
    model = TextClassifier(len(tokenizer.word2idx))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Обучение
    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.LongTensor(X_test)
        test_outputs = model(X_test_tensor)
        predictions = torch.argmax(test_outputs, dim=1).numpy()
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = train_tokenization_model()