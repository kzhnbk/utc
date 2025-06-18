import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    pipeline
)
import pandas as pd

class T5Summarizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def summarize(self, text, max_length=150, min_length=50):
        # T5 требует префикс для задач
        input_text = f"summarize: {text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                length_penalty=2.0
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(self, texts, batch_size=4, max_length=150, min_length=50):
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Добавляем префикс для T5
            input_texts = [f"summarize: {text}" for text in batch_texts]
            
            inputs = self.tokenizer(
                input_texts,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=2.0
                )
            
            batch_summaries = []
            for output in outputs:
                summary = self.tokenizer.decode(output, skip_special_tokens=True)
                batch_summaries.append(summary)
            
            summaries.extend(batch_summaries)
            
        return summaries

class BARTSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def summarize(self, text, max_length=150, min_length=50):
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                length_penalty=2.0
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(self, texts, batch_size=4, max_length=150, min_length=50):
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=2.0
                )
            
            batch_summaries = []
            for output in outputs:
                summary = self.tokenizer.decode(output, skip_special_tokens=True)
                batch_summaries.append(summary)
            
            summaries.extend(batch_summaries)
            
        return summaries

# Упрощенный подход с pipeline
class PipelineSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.summarizer = pipeline("summarization", model=model_name, device=0 if torch.cuda.is_available() else -1)
    
    def summarize(self, text, max_length=150, min_length=50):
        result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    
    def batch_summarize(self, texts, batch_size=4, max_length=150, min_length=50):
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            results = self.summarizer(batch_texts, max_length=max_length, min_length=min_length, do_sample=False)
            batch_summaries = [result['summary_text'] for result in results]
            summaries.extend(batch_summaries)
            
        return summaries

def fine_tune_summarization_model(train_df, model_name='t5-small', model_type='t5'):
    """Дообучение модели для резюмирования"""
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    
    class SummarizationDataset(Dataset):
        def __init__(self, df, tokenizer, model_type='t5', max_input_length=1024, max_target_length=150):
            self.df = df
            self.tokenizer = tokenizer
            self.model_type = model_type
            self.max_input_length = max_input_length
            self.max_target_length = max_target_length
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            text = row['text']
            summary = row['summary']
            
            # Форматирование для T5
            if self.model_type == 't5':
                input_text = f"summarize: {text}"
            else:
                input_text = text
            
            # Токенизация входа
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Токенизация целевого текста
            target_encoding = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten()
            }
    
    if model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:  # BART
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    
    dataset = SummarizationDataset(train_df, tokenizer, model_type)
    
    training_args = TrainingArguments(
        output_dir=f'./{model_type}_summarization_model',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=1000,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()
    
    return model, tokenizer

def ensemble_summarization(texts, models=['t5-small', 'facebook/bart-base']):
    """Ансамбль моделей для лучшего качества"""
    summarizers = []
    
    for model_name in models:
        if 't5' in model_name:
            summarizers.append(T5Summarizer(model_name))
        else:
            summarizers.append(BARTSummarizer(model_name))
    
    ensemble_summaries = []
    
    for text in texts:
        summaries = []
        for summarizer in summarizers:
            summary = summarizer.summarize(text)
            summaries.append(summary)
        
        # Простое объединение - можно улучшить
        ensemble_summary = max(summaries, key=len)  # Выбираем самое длинное
        ensemble_summaries.append(ensemble_summary)
    
    return ensemble_summaries

if __name__ == "__main__":
    # Загрузка данных
    test_df = pd.read_csv('test.csv')  # Колонка 'text'
    
    # Вариант 1: T5
    t5_summarizer = T5Summarizer('t5-small')
    t5_summaries = t5_summarizer.batch_summarize(test_df['text'].tolist())
    
    # Вариант 2: BART
    bart_summarizer = BARTSummarizer('facebook/bart-base')  # bart-base для скорости
    bart_summaries = bart_summarizer.batch_summarize(test_df['text'].tolist())
    
    # Вариант 3: Pipeline (самый простой)
    pipeline_summarizer = PipelineSummarizer('facebook/bart-base')
    pipeline_summaries = pipeline_summarizer.batch_summarize(test_df['text'].tolist())
    
    # Сохранение результатов
    submission_t5 = pd.DataFrame({
        'id': range(len(t5_summaries)),
        'summary': t5_summaries
    })
    submission_t5.to_csv('submission_t5.csv', index=False)
    
    submission_bart = pd.DataFrame({
        'id': range(len(bart_summaries)),
        'summary': bart_summaries
    })
    submission_bart.to_csv('submission_bart.csv', index=False)
    
    # Пример использования
    sample_text = """
    The rapid development of artificial intelligence has transformed various industries. 
    Machine learning algorithms are now capable of processing vast amounts of data and making 
    predictions with remarkable accuracy. Deep learning, a subset of machine learning, has 
    revolutionized fields such as computer vision, natural language processing, and speech 
    recognition. Companies are investing heavily in AI research and development to gain 
    competitive advantages. However, concerns about job displacement, privacy, and ethical 
    implications continue to grow. The future success of AI will depend on addressing these 
    challenges while maximizing the benefits for society.
    """
    
    t5_sum = t5_summarizer.summarize(sample_text)
    bart_sum = bart_summarizer.summarize(sample_text)
    
    print("T5 Summary:", t5_sum)
    print("BART Summary:", bart_sum)