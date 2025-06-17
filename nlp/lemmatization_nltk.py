import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SimpleLemmatizer:
    """Простая лемматизация на основе правил для английского"""
    
    def __init__(self):
        # Базовые правила для английского
        self.rules = [
            (r'ies$', 'y'),      # flies -> fly
            (r'ied$', 'y'),      # carried -> carry
            (r'ying$', 'y'),     # carrying -> carry
            (r'ing$', ''),       # running -> run
            (r'ly$', ''),        # quickly -> quick
            (r'ed$', ''),        # walked -> walk
            (r'ies$', 'y'),      # flies -> fly
            (r'ied$', 'y'),      # died -> die
            (r'ies$', 'y'),      # tries -> try
            (r'es$', ''),        # boxes -> box
            (r's$', ''),         # cats -> cat
        ]
        
        # Неправильные глаголы
        self.irregular = {
            'went': 'go', 'gone': 'go', 'going': 'go',
            'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
            'had': 'have', 'has': 'have', 'having': 'have',
            'did': 'do', 'done': 'do', 'doing': 'do',
            'said': 'say', 'saying': 'say',
            'got': 'get', 'getting': 'get',
            'came': 'come', 'coming': 'come',
            'saw': 'see', 'seen': 'see', 'seeing': 'see',
            'took': 'take', 'taken': 'take', 'taking': 'take',
            'made': 'make', 'making': 'make',
            'knew': 'know', 'known': 'know', 'knowing': 'know',
            'thought': 'think', 'thinking': 'think',
            'found': 'find', 'finding': 'find',
            'gave': 'give', 'given': 'give', 'giving': 'give',
            'told': 'tell', 'telling': 'tell',
            'became': 'become', 'becoming': 'become',
            'left': 'leave', 'leaving': 'leave',
            'felt': 'feel', 'feeling': 'feel',
            'kept': 'keep', 'keeping': 'keep',
            'brought': 'bring', 'bringing': 'bring',
            'began': 'begin', 'begun': 'begin', 'beginning': 'begin',
        }
    
    def lemmatize(self, word):
        word = word.lower()
        
        # Проверяем неправильные формы
        if word in self.irregular:
            return self.irregular[word]
        
        # Применяем правила
        for pattern, replacement in self.rules:
            if re.search(pattern, word):
                return re.sub(pattern, replacement, word)
        
        return word
    
    def lemmatize_text(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return ' '.join([self.lemmatize(word) for word in words])

def preprocess_with_lemmatization():
    # Пример данных
    data = pd.DataFrame({
        'text': [
            'I am running quickly to the store',
            'The cats were playing with toys',
            'She has been working hard lately',
            'They went to the movies yesterday',
            'The children are learning new things'
        ],
        'label': [0, 1, 0, 1, 0]
    })
    
    lemmatizer = SimpleLemmatizer()
    
    # Применяем лемматизацию
    data['lemmatized'] = data['text'].apply(lemmatizer.lemmatize_text)
    
    print("Оригинальный текст vs Лемматизированный:")
    for i, row in data.iterrows():
        print(f"Orig: {row['text']}")
        print(f"Lemm: {row['lemmatized']}")
        print()
    
    # Векторизация с TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    
    # Сравнение с лемматизацией и без
    X_orig = vectorizer.fit_transform(data['text'])
    X_lemm = vectorizer.fit_transform(data['lemmatized'])
    
    y = data['label'].values
    
    # Модели
    model_orig = LogisticRegression()
    model_lemm = LogisticRegression()
    
    if len(data) > 2:  # Если данных достаточно
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X_orig, y, test_size=0.3, random_state=42
        )
        X_train_lemm, X_test_lemm, _, _ = train_test_split(
            X_lemm, y, test_size=0.3, random_state=42
        )
        
        model_orig.fit(X_train_orig, y_train)
        model_lemm.fit(X_train_lemm, y_train)
        
        pred_orig = model_orig.predict(X_test_orig)
        pred_lemm = model_lemm.predict(X_test_lemm)
        
        print("Без лемматизации:")
        print(classification_report(y_test, pred_orig))
        print("С лемматизацией:")
        print(classification_report(y_test, pred_lemm))
    
    return lemmatizer, vectorizer, data

if __name__ == "__main__":
    lemmatizer, vectorizer, processed_data = preprocess_with_lemmatization()