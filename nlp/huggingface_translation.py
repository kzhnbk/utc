import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import numpy as np

class T5TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Префикс для T5
        input_text = f"translate English to Russian: {row['source']}"
        target_text = row['target']
        
        # Токенизация
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def train_t5_translation():
    # Загрузка модели
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Пример данных
    data = {
        'source': ['Hello world', 'How are you?', 'Good morning', 'Thank you'] * 50,
        'target': ['Привет мир', 'Как дела?', 'Доброе утро', 'Спасибо'] * 50
    }
    
    df = pd.DataFrame(data)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Создание датасетов
    train_dataset = T5TranslationDataset(train_df, tokenizer)
    val_dataset = T5TranslationDataset(val_df, tokenizer)
    
    # Настройка обучения
    training_args = TrainingArguments(
        output_dir='./t5_translation',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=500,
        eval_steps=500,
        evaluation_strategy='steps',
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    # Обучение
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    # Сохранение
    model.save_pretrained('./t5_translation_final')
    tokenizer.save_pretrained('./t5_translation_final')
    
    return model, tokenizer

def generate_translation(model, tokenizer, text):
    input_text = f"translate English to Russian: {text}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Создание submission файла для Kaggle
def create_submission(model, tokenizer, test_df, output_path='submission.csv'):
    predictions = []
    
    for _, row in test_df.iterrows():
        translation = generate_translation(model, tokenizer, row['source'])
        predictions.append(translation)
    
    submission_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'translation': predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    model, tokenizer = train_t5_translation()
    
    # Тест
    test_text = "Hello, how are you today?"
    result = generate_translation(model, tokenizer, test_text)
    print(f"Translation: {result}")