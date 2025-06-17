import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset

class TextGenerationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Добавляем специальные токены
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_gpt2_generation():
    # Загрузка модели
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Добавляем pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Пример данных для обучения (замените на реальные)
    texts = [
        "Artificial intelligence is transforming the world in many ways.",
        "Machine learning algorithms can learn from data and make predictions.",
        "Deep learning uses neural networks to solve complex problems.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information."
    ] * 100
    
    # Создание датасета
    dataset = TextGenerationDataset(texts, tokenizer)
    
    # Настройка обучения
    training_args = TrainingArguments(
        output_dir='./gpt2_generation',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        logging_steps=100,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Обучение
    trainer.train()
    
    # Сохранение
    model.save_pretrained('./gpt2_generation_final')
    tokenizer.save_pretrained('./gpt2_generation_final')
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, num_return_sequences=1):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones(inputs.shape, dtype=torch.long)
        )
    
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

def create_generation_submission(model, tokenizer, test_prompts, output_path='submission.csv'):
    results = []
    
    for i, prompt in enumerate(test_prompts):
        generated = generate_text(model, tokenizer, prompt, max_length=150)
        results.append({
            'id': i,
            'prompt': prompt,
            'generated_text': generated[0]
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Generation submission saved to {output_path}")

if __name__ == "__main__":
    model, tokenizer = train_gpt2_generation()
    
    # Тест генерации
    prompt = "The future of artificial intelligence"
    generated = generate_text(model, tokenizer, prompt)
    print(f"Generated: {generated[0]}")