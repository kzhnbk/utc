import torch
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class SummarizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length=512, max_target_length=128):
        self.data = df
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Исходный текст и цель
        input_text = row['text']
        target_text = row['summary']
        
        # Токенизация входа
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Токенизация цели
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def train_bart_summarization():
    # Загрузка модели
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Пример данных
    texts = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents.",
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning."
    ] * 50
    
    summaries = [
        "AI is machine intelligence vs natural intelligence",
        "ML automates analytical model building using data",
        "Deep learning uses neural networks for representation learning"
    ] * 50
    
    data = pd.DataFrame({
        'text': texts,
        'summary': summaries
    })
    
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
    
    # Создание датасетов
    train_dataset = SummarizationDataset(train_df, tokenizer)
    val_dataset = SummarizationDataset(val_df, tokenizer)
    
    # Настройка обучения
    training_args = TrainingArguments(
        output_dir='./bart_summarization',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=500,
        eval_steps=500,
        evaluation_strategy='steps',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Обучение
    trainer.train()
    
    # Сохранение
    model.save_pretrained('./bart_summarization_final')
    tokenizer.save_pretrained('./bart_summarization_final')
    
    return model, tokenizer

def generate_summary(model, tokenizer, text, max_length=128):
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            min_length=20,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def create_summarization_submission(model, tokenizer, test_df, output_path='submission.csv'):
    summaries = []
    
    for _, row in test_df.iterrows():
        summary = generate_summary(model, tokenizer, row['text'])
        summaries.append(summary)
    
    submission_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'summary': summaries
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Summarization submission saved to {output_path}")

# ROUGE метрики для оценки
def calculate_rouge_scores(predictions, references):
    from sklearn.metrics import accuracy_score
    import re
    
    def get_tokens(text):
        return re.findall(r'\w+', text.lower())
    
    rouge_1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(get_tokens(pred))
        ref_tokens = set(get_tokens(ref))
        
        if len(ref_tokens) == 0:
            rouge_1_scores.append(0)
        else:
            overlap = len(pred_tokens.intersection(ref_tokens))
            rouge_1 = overlap / len(ref_tokens)
            rouge