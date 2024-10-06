from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import google.cloud.aiplatform as vertex_ai

# Load dataset from Google Cloud Storage
dataset = load_dataset('csv', data_files='gs://review-creation/dataset/reviews_supplements.csv')

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/tmp/results',           # Output directory for model
    overwrite_output_dir=True,
    num_train_epochs=3,                  # Number of training epochs
    per_device_train_batch_size=4,       # Batch size per device
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train']
)

# Train the model
trainer.train()

# Save the fine-tuned model to Google Cloud Storage
trainer.save_model('gs://your-bucket-name/fine_tuned_gpt2/')
tokenizer.save_pretrained('gs://your-bucket-name/fine_tuned_gpt2/')
