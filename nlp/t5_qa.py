import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

class T5QA:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def predict(self, question, context, max_length=100):
        # Форматируем входные данные для T5
        input_text = f"question: {question} context: {context}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def batch_predict(self, questions, contexts, batch_size=4):
        predictions = []
        
        for i in range(0, len(questions), batch_size):
            batch_q = questions[i:i+batch_size]
            batch_c = contexts[i:i+batch_size]
            
            # Подготавливаем batch
            input_texts = [f"question: {q} context: {c}" for q, c in zip(batch_q, batch_c)]
            
            inputs = self.tokenizer(
                input_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            batch_answers = []
            for output in outputs:
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                batch_answers.append(answer)
            
            predictions.extend(batch_answers)
            
        return predictions

def fine_tune_t5_qa(train_df, model_name='t5-small', epochs=3):
    """Дообучение T5 для QA"""
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    
    class T5QADataset(Dataset):
        def __init__(self, df, tokenizer, max_input_length=512, max_target_length=100):
            self.df = df
            self.tokenizer = tokenizer
            self.max_input_length = max_input_length
            self.max_target_length = max_target_length
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            question = row['question']
            context = row['context']
            answer = row['answer']
            
            # Форматируем для T5
            input_text = f"question: {question} context: {context}"
            target_text = answer
            
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
                target_text,
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
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    dataset = T5QADataset(train_df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./t5_qa_model',
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=1000,
        eval_steps=1000,
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

def predict_with_t5_qa(test_df, model_path=None):
    if model_path:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        qa_model = T5QA()
        qa_model.tokenizer = tokenizer
        qa_model.model = model
    else:
        qa_model = T5QA()
    
    questions = test_df['question'].tolist()
    contexts = test_df['context'].tolist()
    
    predictions = qa_model.batch_predict(questions, contexts)
    
    return predictions

if __name__ == "__main__":
    # Загрузка данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Предсказания с предобученной моделью
    predictions = predict_with_t5_qa(test_df)
    
    # Или дообучение модели
    # model, tokenizer = fine_tune_t5_qa(train_df)
    # predictions = predict_with_t5_qa(test_df, './t5_qa_model')
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'answer': predictions
    })
    submission.to_csv('submission_t5_qa.csv', index=False)