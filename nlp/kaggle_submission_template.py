import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import bleu_score
import matplotlib.pyplot as plt

class KaggleSubmissionHandler:
    def __init__(self, model, tokenizer, task_type='translation'):
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type
        
    def load_test_data(self, test_path):
        """Загрузка тестовых данных"""
        if test_path.endswith('.csv'):
            return pd.read_csv(test_path)
        elif test_path.endswith('.json'):
            return pd.read_json(test_path)
        else:
            raise ValueError("Unsupported file format")
    
    def preprocess_for_translation(self, test_df):
        """Предобработка для задач перевода"""
        if 'source' not in test_df.columns:
            raise ValueError("Test data must have 'source' column")
        
        processed_texts = []
        for text in test_df['source']:
            # Базовая очистка
            text = str(text).strip()
            processed_texts.append(text)
        
        return processed_texts
    
    def preprocess_for_generation(self, test_df):
        """Предобработка для задач генерации"""
        if 'prompt' not in test_df.columns:
            raise ValueError("Test data must have 'prompt' column")
        
        processed_prompts = []
        for prompt in test_df['prompt']:
            prompt = str(prompt).strip()
            processed_prompts.append(prompt)
        
        return processed_prompts
    
    def predict_translation(self, texts, max_length=128, batch_size=8):
        """Предсказание перевода"""
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Для T5 добавляем префикс
            if 't5' in self.tokenizer.name_or_path.lower():
                batch_texts = [f"translate English to Russian: {text}" for text in batch_texts]
            
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            batch_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def predict_generation(self, prompts, max_length=100, batch_size=4):
        """Предсказание генерации текста"""
        predictions = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            batch_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def create_submission(self, test_path, output_path='submission.csv'):
        """Создание файла для submission"""
        test_df = self.load_test_data(test_path)
        
        if self.task_type == 'translation':
            texts = self.preprocess_for_translation(test_df)
            predictions = self.predict_translation(texts)
            
            submission_df = pd.DataFrame({
                'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
                'translation': predictions
            })
            
        elif self.task_type == 'generation':
            prompts = self.preprocess_for_generation(test_df)
            predictions = self.predict_generation(prompts)
            
            submission_df = pd.DataFrame({
                'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
                'generated_text': predictions
            })
            
        elif self.task_type == 'summarization':
            texts = test_df['text'].tolist()
            predictions = self.predict_translation(texts, max_length=64)  # Короче для summary
            
            submission_df = pd.DataFrame({
                'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
                'summary': predictions
            })
        
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Submission shape: {submission_df.shape}")
        
        return submission_df

def evaluate_predictions(predictions, references, task_type='translation'):
    """Простая оценка предсказаний"""
    if task_type == 'translation':
        # BLEU score (упрощенный)
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # Простой 1-gram BLEU
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            
            if len(ref_set) > 0:
                precision = len(pred_set.intersection(ref_set)) / len(pred_set) if len(pred_set) > 0 else 0
                bleu_scores.append(precision)
            else:
                bleu_scores.append(0)
        
        return np.mean(bleu_scores)
    
    elif task_type == 'generation':
        # Средняя длина
        avg_length = np.mean([len(pred.split()) for pred in predictions])
        return avg_length

# Пример использования для разных моделей
def create_translation_submission():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    handler = KaggleSubmissionHandler(model, tokenizer, task_type='translation')
    
    # Создание тестовых данных (замените на реальные)
    test_data = pd.DataFrame({
        'id': range(100),
        'source': ['Hello world', 'How are you?'] * 50
    })
    test_data.to_csv('test.csv', index=False)
    
    # Создание submission
    submission = handler.create_submission('test.csv', 'translation_submission.csv')
    return submission

def create_generation_submission():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    handler = KaggleSubmissionHandler(model, tokenizer, task_type='generation')
    
    # Создание тестовых данных
    test_data = pd.DataFrame({
        'id': range(50),
        'prompt': ['The future of AI', 'Machine learning is'] * 25
    })
    test_data.to_csv('test_generation.csv', index=False)
    
    # Создание submission
    submission = handler.create_submission('test_generation.csv', 'generation_submission.csv')
    return submission

if __name__ == "__main__":
    print("Creating translation submission...")
    trans_sub = create_translation_submission()
    
    print("\nCreating generation submission...")
    gen_sub = create_generation_submission()
    
    print("Submissions created successfully!")