import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class BERTFeatureExtractor:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract_features(self, texts, max_length=128):
        features = []
        
        for text in texts:
            # Токенизация
            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
            
            # Извлечение признаков
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем [CLS] токен как представление предложения
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(cls_embedding.flatten())
        
        return np.array(features)

class CustomTokenizationBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
    def analyze_tokenization(self, texts):
        results = []
        
        for text in texts:
            # Разные виды токенизации
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Полная токенизация
            encoded = self.tokenizer(text, return_tensors='pt')
            
            result = {
                'original': text,
                'tokens': tokens,
                'token_ids': token_ids,
                'attention_mask': encoded['attention_mask'].tolist(),
                'input_ids': encoded['input_ids'].tolist(),
                'num_tokens': len(tokens)
            }
            results.append(result)
            
        return results

def bert_classification_pipeline():
    # Пример данных
    data = pd.DataFrame({
        'text': [
            'This movie is absolutely fantastic!',
            'Terrible acting and boring plot.',
            'Great storyline and amazing visuals.',
            'Worst movie I have ever seen.',
            'Perfect blend of action and drama.',
            'Not worth watching at all.'
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    })
    
    print("Анализ токенизации BERT:")
    tokenization_analyzer = CustomTokenizationBERT()
    token_analysis = tokenization_analyzer.analyze_tokenization(data['text'].tolist())
    
    for analysis in token_analysis[:2]:  # Показываем первые 2
        print(f"Текст: {analysis['original']}")
        print(f"Токены: {analysis['tokens']}")
        print(f"Количество токенов: {analysis['num_tokens']}")
        print()
    
    # Извлечение признаков с BERT
    print("Извлечение признаков с BERT...")
    feature_extractor = BERTFeatureExtractor()
    X = feature_extractor.extract_features(data['text'].tolist())
    y = data['label'].values
    
    print(f"Размер признаков: {X.shape}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Обучение классификатора
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Предсказания
    y_pred = classifier.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return feature_extractor, classifier, data

def compare_tokenizers():
    """Сравнение разных токенизаторов"""
    text = "The running dogs are quickly chasing the cats in the beautiful garden."
    
    # BERT токенизатор
    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert_tokens = bert_tokenizer.tokenize(text)
    
    # Простая токенизация по словам
    simple_tokens = text.lower().split()
    
    # Токенизация по символам
    char_tokens = list(text.lower().replace(' ', '_'))
    
    print("Сравнение токенизаторов:")
    print(f"Оригинал: {text}")
    print(f"BERT токены ({len(bert_tokens)}): {bert_tokens}")
    print(f"Простые токены ({len(simple_tokens)}): {simple_tokens}")
    print(f"Символьные токены ({len(char_tokens)}): {char_tokens[:20]}...")  # Первые 20
    
    return {
        'bert': bert_tokens,
        'simple': simple_tokens,
        'char': char_tokens
    }

if __name__ == "__main__":
    # Запуск основного пайплайна
    feature_extractor, classifier, data = bert_classification_pipeline()
    
    # Сравнение токенизаторов
    tokenizer_comparison = compare_tokenizers()