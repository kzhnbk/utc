import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import math

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        
    def fit(self, corpus):
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        
        # Токенизация и подсчет статистик
        tokenized_corpus = [self.tokenize(doc) for doc in corpus]
        
        # Средняя длина документа
        self.avgdl = sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus)
        
        # Частоты терминов в документах
        for doc in tokenized_corpus:
            self.doc_lens.append(len(doc))
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            
        # IDF расчет
        df = Counter()
        for doc in tokenized_corpus:
            for term in set(doc):
                df[term] += 1
                
        for term, freq in df.items():
            self.idf[term] = math.log(len(corpus) / freq)
    
    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def score(self, query, doc_idx):
        query_tokens = self.tokenize(query)
        doc_freqs = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0
        for term in query_tokens:
            if term in doc_freqs:
                tf = doc_freqs[term]
                idf = self.idf.get(term, 0)
                
                # BM25 формула
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
                
        return score
    
    def search(self, query, top_k=5):
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Пример использования для QA системы
def qa_with_bm25(train_df, test_df):
    """
    train_df: columns ['context', 'question', 'answer']
    test_df: columns ['context', 'question']
    """
    
    # Группируем по контексту
    contexts = train_df['context'].unique()
    
    predictions = []
    
    for _, row in test_df.iterrows():
        question = row['question']
        context = row['context']
        
        # Находим релевантные предложения в контексте
        sentences = context.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # BM25 поиск
        bm25 = BM25()
        bm25.fit(sentences)
        
        # Ищем наиболее релевантные предложения
        results = bm25.search(question, top_k=3)
        
        # Объединяем топ предложения как ответ
        answer_sentences = [sentences[idx] for idx, _ in results]
        answer = '. '.join(answer_sentences)
        
        predictions.append(answer)
    
    return predictions

# Загрузка и обработка данных
if __name__ == "__main__":
    # Пример загрузки данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    predictions = qa_with_bm25(train_df, test_df)
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'answer': predictions
    })
    submission.to_csv('submission_bm25.csv', index=False)