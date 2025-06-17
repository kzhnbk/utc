import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Скачиваем нужные данные NLTK (запускать один раз)
# nltk.download('punkt')
# nltk.download('stopwords')

class TextRankSummarizer:
    def __init__(self, language='english'):
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()
        
    def preprocess_text(self, text):
        """Предобработка текста"""
        # Убираем лишние символы
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    def sentence_similarity(self, sent1, sent2):
        """Вычисляем схожесть между предложениями"""
        # Токенизация
        words1 = word_tokenize(sent1.lower())
        words2 = word_tokenize(sent2.lower())
        
        # Убираем стоп-слова
        words1 = [w for w in words1 if w not in self.stop_words]
        words2 = [w for w in words2 if w not in self.stop_words]
        
        # Объединяем словари
        all_words = list(set(words1 + words2))
        
        # Создаем векторы
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        for w in words1:
            if w in all_words:
                vector1[all_words.index(w)] += 1
                
        for w in words2:
            if w in all_words:
                vector2[all_words.index(w)] += 1
        
        # Косинусное сходство
        return self.cosine_similarity(vector1, vector2)
    
    def cosine_similarity(self, vec1, vec2):
        """Косинусное сходство между векторами"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)
    
    def build_similarity_matrix(self, sentences):
        """Строим матрицу схожести предложений"""
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(sentences[i], sentences[j])
        
        return similarity_matrix
    
    def textrank(self, text, num_sentences=3):
        """Основная функция TextRank"""
        # Разбиваем на предложения
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Строим матрицу схожести
        similarity_matrix = self.build_similarity_matrix(sentences)
        
        # Создаем граф
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Применяем PageRank
        scores = nx.pagerank(graph)
        
        # Сортируем предложения по рангу
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        # Выбираем топ предложения
        top_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
        
        # Возвращаем в исходном порядке
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)

class TfidfTextRank:
    """Улучшенная версия с TF-IDF"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    def summarize(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Создаем TF-IDF матрицу
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Вычисляем косинусное сходство
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Создаем граф
        graph = nx.from_numpy_array(similarity_matrix)
        
        # PageRank
        scores = nx.pagerank(graph)
        
        # Ранжируем предложения
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        # Выбираем топ предложения в исходном порядке
        top_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
        
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)

def summarize_with_textrank(df, text_column='text', summary_length=3):
    """Функция для обработки датасета"""
    summarizer = TfidfTextRank()
    
    summaries = []
    for _, row in df.iterrows():
        text = row[text_column]
        summary = summarizer.summarize(text, num_sentences=summary_length)
        summaries.append(summary)
    
    return summaries

if __name__ == "__main__":
    # Загрузка данных
    test_df = pd.read_csv('test.csv')  # Колонка 'text' с текстами для резюмирования
    
    # Генерируем резюме
    summaries = summarize_with_textrank(test_df, text_column='text', summary_length=3)
    
    # Сохраняем результаты
    submission = pd.DataFrame({
        'id': range(len(summaries)),
        'summary': summaries
    })
    submission.to_csv('submission_textrank.csv', index=False)
    
    # Пример использования
    sample_text = """
    Artificial intelligence is transforming the world. Machine learning algorithms are becoming more sophisticated. 
    Deep learning has revolutionized computer vision and natural language processing. 
    AI applications are found in healthcare, finance, and transportation. 
    However, there are concerns about job displacement and privacy. 
    The future of AI depends on responsible development and deployment.
    """
    
    summarizer = TextRankSummarizer()
    summary = summarizer.textrank(sample_text, num_sentences=2)
    print("Summary:", summary)