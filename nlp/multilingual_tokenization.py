import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class MultilingualProcessor:
    """Многоязычная обработка текста"""
    
    def __init__(self):
        # Базовые правила лемматизации для разных языков
        self.lemma_rules = {
            'en': [  # Английский
                (r'ing$', ''), (r'ed$', ''), (r'ly$', ''), 
                (r'ies$', 'y'), (r'ied$', 'y'), (r'es$', ''), (r's$', '')
            ],
            'ru': [  # Русский (упрощенные правила)
                (r'ами$', 'а'), (r'ах$', 'а'), (r'ой$', 'а'), (r'ы$', 'а'),
                (r'ами$', 'ы'), (r'ах$', 'ы'), (r'ов$', ''), (r'ами$', ''),
                (r'ить$', 'ить'), (r'ать$', 'ать'), (r'еть$', 'еть')
            ],
            'es': [  # Испанский
                (r'ando$', 'ar'), (r'iendo$', 'er'), (r'ado$', 'ar'),
                (r'ido$', 'er'), (r'os$', 'o'), (r'as$', 'a'), (r'es$', '')
            ]
        }
        
        self.stop_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'ru': {'и', 'в', 'на', 'с', 'по', 'за', 'о', 'к', 'от', 'до', 'для', 'как', 'что', 'это', 'не'},
            'es': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da'}
        }
    
    def detect_language(self, text):
        """Простое определение языка по символам"""
        cyrillic_count = len(re.findall(r'[а-яё]', text.lower()))
        spanish_chars = len(re.findall(r'[ñáéíóúü]', text.lower()))
        total_chars = len(re.findall(r'[a-zA-Zа-яёñáéíóúü]', text.lower()))
        
        if total_chars == 0:
            return 'en'
        
        if cyrillic_count / total_chars > 0.3:
            return 'ru'
        elif spanish_chars > 0:
            return 'es'
        else:
            return 'en'
    
    def simple_lemmatize(self, word, lang):
        """Простая лемматизация по правилам"""
        if lang not in self.lemma_rules:
            return word
            
        for pattern, replacement in self.lemma_rules[lang]:
            if re.search(pattern, word):
                return re.sub(pattern, replacement, word)
        return word
    
    def tokenize_and_lemmatize(self, text):
        """Токенизация и лемматизация многоязычного текста"""
        lang = self.detect_language(text)
        
        # Токенизация
        if lang == 'ru':
            tokens = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
        elif lang == 'es':
            tokens = re.findall(r'[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+', text.lower())
        else:  # en
            tokens = re.findall(r'[a-zA-Z]+', text.lower())
        
        # Удаление стоп-слов и лемматизация
        stop_words = self.stop_words.get(lang, set())
        processed_tokens = []
        
        for token in tokens:
            if token not in stop_words and len(token) > 2:
                lemma = self.simple_lemmatize(token, lang)
                processed_tokens.append(lemma)
        
        return processed_tokens, lang

class MultilingualBERTClassifier:
    """Многоязычный классификатор на основе BERT"""
    
    def __init__(self, model_name='distilbert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract_features(self, texts, max_length=128):
        """Извлечение признаков из многоязычных текстов"""
        features = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(cls_embedding.flatten())
        
        return np.array(features)

def multilingual_classification_pipeline():
    """Пайплайн для многоязычной классификации"""
    
    # Многоязычные данные
    data = pd.DataFrame({
        'text': [
            'This is a great product!',  # en
            'Este producto es excelente!',  # es
            'Этот продукт просто великолепен!',  # ru
            'Terrible quality, very disappointed.',  # en
            'Calidad terrible, muy decepcionado.',  # es
            'Ужасное качество, очень разочарован.',  # ru
            'Amazing service and fast delivery!',  # en
            'Servicio increíble y entrega rápida!',  # es
            'Потрясающий сервис и быстрая доставка!',  # ru
        ],
        'label': [1, 1, 1, 0, 0, 0, 1, 1, 1]  # 1=positive, 0=negative
    })
    
    # Обработка с помощью простого процессора
    processor = MultilingualProcessor()
    
    print("Анализ многоязычного текста:")
    for i, row in data.iterrows():
        tokens, lang = processor.tokenize_and_lemmatize(row['text'])
        print(f"Текст: {row['text']}")
        print(f"Язык: {lang}, Токены: {tokens}")
        print()
    
    # Классификация с многоязычным BERT
    print("Обучение многоязычного классификатора...")
    bert_classifier = MultilingualBERTClassifier()
    
    X = bert_classifier.extract_features(data['text'].tolist())
    y = data['label'].values
    
    print(f"Размер признаков: {X.shape}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Обучение классификатора
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Предсказания
    y_pred = classifier.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return processor, bert_classifier, classifier

def analyze_tokenization_differences():
    """Анализ различий в токенизации для разных языков"""
    
    texts = {
        'en': "The quick brown fox jumps over the lazy dog",
        'ru': "Быстрая коричневая лиса прыгает через ленивую собаку",
        'es': "El rápido zorro marrón salta sobre el perro perezoso"
    }
    
    processor = MultilingualProcessor()
    
    # Многоязычный BERT токенизатор
    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    
    print("Сравнение токенизации для разных языков:")
    
    for lang, text in texts.items():
        # Наша токенизация
        our_tokens, detected_lang = processor.tokenize_and_lemmatize(text)
        
        # BERT токенизация
        bert_tokens = bert_tokenizer.tokenize(text)
        
        print(f"\n{lang.upper()} ({detected_lang}):")
        print(f"Текст: {text}")
        print(f"Наши токены ({len(our_tokens)}): {our_tokens}")
        print(f"BERT токены ({len(bert_tokens)}): {bert_tokens}")
    
    return texts

class AdvancedTokenizer:
    """Продвинутый токенизатор с обработкой специальных случаев"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'hashtag': r'#\w+',
            'mention': r'@\w+',
            'number': r'\b\d+(?:\.\d+)?\b'
        }
    
    def extract_special_tokens(self, text):
        """Извлечение специальных токенов"""
        special_tokens = {}
        
        for token_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                special_tokens[token_type] = matches
        
        return special_tokens
    
    def advanced_tokenize(self, text):
        """Продвинутая токенизация с обработкой специальных случаев"""
        # Извлекаем специальные токены
        special_tokens = self.extract_special_tokens(text)
        
        # Заменяем специальные токены плейсхолдерами
        processed_text = text
        replacements = {}
        
        for token_type, tokens in special_tokens.items():
            for i, token in enumerate(tokens):
                placeholder = f"__{token_type.upper()}{i}__"
                processed_text = processed_text.replace(token, placeholder)
                replacements[placeholder] = token
        
        # Обычная токенизация
        regular_tokens = re.findall(r'\b\w+\b', processed_text.lower())
        
        # Восстанавливаем специальные токены
        final_tokens = []
        for token in regular_tokens:
            if token in replacements:
                final_tokens.append(replacements[token])
            else:
                final_tokens.append(token)
        
        return final_tokens, special_tokens

def test_advanced_tokenization():
    """Тестирование продвинутой токенизации"""
    
    text = """
    Contact us at info@example.com or visit https://www.example.com
    Call 123-456-7890 for support. Follow us @company #AI #ML
    Price: $29.99 per month.
    """
    
    tokenizer = AdvancedTokenizer()
    tokens, special_tokens = tokenizer.advanced_tokenize(text)
    
    print("Продвинутая токенизация:")
    print(f"Текст: {text.strip()}")
    print(f"Токены: {tokens}")
    print(f"Специальные токены: {special_tokens}")
    
    return tokenizer

if __name__ == "__main__":
    # Многоязычная классификация
    processor, bert_classifier, classifier = multilingual_classification_pipeline()
    
    # Анализ различий токенизации
    texts = analyze_tokenization_differences()
    
    # Продвинутая токенизация
    advanced_tokenizer = test_advanced_tokenization()