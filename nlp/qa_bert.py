import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
import numpy as np

class BertQA:
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def predict(self, question, context):
        inputs = self.tokenizer(
            question, context,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Находим лучшие позиции начала и конца
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Декодируем ответ
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer
    
    def batch_predict(self, questions, contexts, batch_size=8):
        predictions = []
        
        for i in range(0, len(questions), batch_size):
            batch_q = questions[i:i+batch_size]
            batch_c = contexts[i:i+batch_size]
            
            batch_answers = []
            for q, c in zip(batch_q, batch_c):
                answer = self.predict(q, c)
                batch_answers.append(answer)
            
            predictions.extend(batch_answers)
            
        return predictions

def fine_tune_bert_qa(train_df, model_name='distilbert-base-uncased-distilled-squad'):
    """Дообучение BERT для QA"""
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    
    class QADataset(Dataset):
        def __init__(self, df, tokenizer, max_length=512):
            self.df = df
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            question = row['question']
            context = row['context']
            answer = row['answer']
            
            # Находим позицию ответа в контексте
            start_pos = context.find(answer)
            end_pos = start_pos + len(answer) if start_pos != -1 else 0
            
            encoding = self.tokenizer(
                question, context,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Токенизируем только контекст для поиска позиций
            context_encoding = self.tokenizer(
                context,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True
            )
            
            # Находим токенные позиции начала и конца ответа
            start_token_pos = 0
            end_token_pos = 0
            
            if start_pos != -1:
                for i, (start, end) in enumerate(context_encoding['offset_mapping']):
                    if start <= start_pos < end:
                        start_token_pos = i + len(self.tokenizer(question)['input_ids']) + 1
                    if start < start_pos + len(answer) <= end:
                        end_token_pos = i + len(self.tokenizer(question)['input_ids']) + 1
                        break
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'start_positions': torch.tensor(start_token_pos, dtype=torch.long),
                'end_positions': torch.tensor(end_token_pos, dtype=torch.long)
            }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    dataset = QADataset(train_df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./bert_qa_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
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

# Основная функция для предсказаний
def predict_with_bert_qa(test_df, model_path=None):
    if model_path:
        qa_model = BertQA(model_path)
    else:
        qa_model = BertQA()
    
    questions = test_df['question'].tolist()
    contexts = test_df['context'].tolist()
    
    predictions = qa_model.batch_predict(questions, contexts)
    
    return predictions

if __name__ == "__main__":
    # Загрузка данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Предсказания с предобученной моделью
    predictions = predict_with_bert_qa(test_df)
    
    # Или дообучение собственной модели
    # model, tokenizer = fine_tune_bert_qa(train_df)
    # predictions = predict_with_bert_qa(test_df, './bert_qa_model')
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'answer': predictions
    })
    submission.to_csv('submission_bert_qa.csv', index=False)