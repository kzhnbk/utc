# IOAI Text Classification - Multiple Approaches
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import re
import warnings
warnings.filterwarnings('ignore')

# ============= PREPROCESSING =============
def preprocess_text(text):
    """Basic text preprocessing"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# ============= SCENARIO 1: WORD2VEC + CLASSICAL ML =============
class Word2VecClassifier:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        self.classifier = None
        
    def text_to_vector(self, text):
        """Convert text to average word2vec vector"""
        words = text.split()
        vectors = []
        for word in words:
            if word in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[word])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def fit(self, texts, labels, classifier_type='nb'):
        # Train Word2Vec
        tokenized_texts = [text.split() for text in texts]
        self.w2v_model = Word2Vec(tokenized_texts, vector_size=self.vector_size, 
                                 window=self.window, min_count=self.min_count)
        
        # Convert texts to vectors
        X = np.array([self.text_to_vector(text) for text in texts])
        
        # Train classifier
        if classifier_type == 'nb':
            # For NB, ensure positive values
            X = np.abs(X)
            self.classifier = MultinomialNB()
        else:
            self.classifier = SVC(kernel='rbf')
        
        self.classifier.fit(X, labels)
        
    def predict(self, texts):
        X = np.array([self.text_to_vector(text) for text in texts])
        if isinstance(self.classifier, MultinomialNB):
            X = np.abs(X)
        return self.classifier.predict(X)

# ============= SCENARIO 2: BERT FINE-TUNING =============
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              epochs=3, batch_size=16):
        train_dataset = BERTDataset(train_texts, train_labels, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy='no'  # Don't save to avoid disk issues
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
    def predict(self, texts):
        self.model.eval()
        predictions = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1)
                predictions.append(pred.item())
        
        return np.array(predictions)

# ============= SCENARIO 3: BERT EMBEDDINGS + CLASSICAL ML =============
class BERTEmbeddingClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = None
        
    def get_embeddings(self, texts):
        """Extract BERT embeddings"""
        embeddings = []
        self.model.eval()
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)
    
    def fit(self, texts, labels, classifier_type='svm'):
        X = self.get_embeddings(texts)
        
        if classifier_type == 'nb':
            # Make values positive for NB
            X = np.abs(X)
            self.classifier = MultinomialNB()
        else:
            self.classifier = SVC(kernel='rbf')
        
        self.classifier.fit(X, labels)
        
    def predict(self, texts):
        X = self.get_embeddings(texts)
        if isinstance(self.classifier, MultinomialNB):
            X = np.abs(X)
        return self.classifier.predict(X)

# ============= MAIN EXECUTION FUNCTION =============
def run_classification_experiments(train_df, test_df, text_col='text', label_col='label'):
    """
    Main function to run all classification experiments
    
    Expected DataFrame format:
    - train_df: columns [text_col, label_col]
    - test_df: columns [text_col] (labels for evaluation if available)
    """
    
    print("Preprocessing data...")
    train_texts = train_df[text_col].apply(preprocess_text).tolist()
    train_labels = train_df[label_col].tolist()
    test_texts = test_df[text_col].apply(preprocess_text).tolist()
    
    # Convert labels to numeric if needed
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_labels_numeric = le.fit_transform(train_labels)
    
    results = {}
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts, train_labels_numeric, test_size=0.2, random_state=42)
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: WORD2VEC + NAIVE BAYES")
    print("="*50)
    
    w2v_nb = Word2VecClassifier()
    w2v_nb.fit(X_train, y_train, classifier_type='nb')
    val_pred = w2v_nb.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    test_pred = w2v_nb.predict(test_texts)
    results['Word2Vec_NB'] = test_pred
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: WORD2VEC + SVM")
    print("="*50)
    
    w2v_svm = Word2VecClassifier()
    w2v_svm.fit(X_train, y_train, classifier_type='svm')
    val_pred = w2v_svm.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    test_pred = w2v_svm.predict(test_texts)
    results['Word2Vec_SVM'] = test_pred
    
    print("\n" + "="*50)
    print("EXPERIMENT 3: BERT EMBEDDINGS + SVM")
    print("="*50)
    
    bert_svm = BERTEmbeddingClassifier()
    bert_svm.fit(X_train, y_train, classifier_type='svm')
    val_pred = bert_svm.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    test_pred = bert_svm.predict(test_texts)
    results['BERT_SVM'] = test_pred
    
    print("\n" + "="*50)
    print("EXPERIMENT 4: BERT FINE-TUNING")
    print("="*50)
    
    num_labels = len(np.unique(train_labels_numeric))
    bert_ft = BERTClassifier(num_labels=num_labels)
    bert_ft.train(X_train, y_train, epochs=2)  # Reduced epochs for speed
    
    val_pred = bert_ft.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    test_pred = bert_ft.predict(test_texts)
    results['BERT_FineTuned'] = test_pred
    
    # Create submission files
    print("\nCreating submission files...")
    for method, predictions in results.items():
        # Convert back to original labels
        original_predictions = le.inverse_transform(predictions)
        
        submission_df = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': original_predictions
        })
        submission_df.to_csv(f'submission_{method}.csv', index=False)
        print(f"Saved: submission_{method}.csv")
    
    return results

# ============= DEMO WITH SYNTHETIC DATA =============
def create_demo_data():
    """Create demo data for testing"""
    np.random.seed(42)
    
    # Synthetic text data
    positive_texts = [
        "This movie is amazing and wonderful",
        "Great performance by the actors",
        "Loved every moment of this film",
        "Excellent story and direction",
        "Beautiful cinematography and music"
    ] * 20
    
    negative_texts = [
        "This movie is terrible and boring",
        "Poor acting and bad script",
        "Waste of time and money",
        "Horrible direction and editing",
        "Very disappointing experience"
    ] * 20
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Split train/test
    split_idx = int(0.8 * len(texts))
    
    train_df = pd.DataFrame({
        'text': texts[:split_idx],
        'label': labels[:split_idx]
    })
    
    test_df = pd.DataFrame({
        'text': texts[split_idx:],
        'label': labels[split_idx:]  # Include for evaluation
    })
    
    return train_df, test_df

# ============= USAGE EXAMPLE =============
if __name__ == "__main__":
    print("Creating demo data...")
    train_df, test_df = create_demo_data()
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")
    
    # Run experiments
    results = run_classification_experiments(train_df, test_df)
    
    print("\nExperiments completed!")
    print("Check the generated submission_*.csv files")

# ============= ALTERNATIVE: TF-IDF BASELINE =============
def tfidf_baseline(train_df, test_df, text_col='text', label_col='label'):
    """Quick TF-IDF baseline"""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, train_df[label_col])
    nb_pred = nb.predict(X_test)
    
    # SVM
    svm = SVC()
    svm.fit(X_train, train_df[label_col])
    svm_pred = svm.predict(X_test)
    
    return {'TF-IDF_NB': nb_pred, 'TF-IDF_SVM': svm_pred}