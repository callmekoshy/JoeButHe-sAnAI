from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load your data from train_data.txt
with open("train_data.txt", "r", encoding="utf-8") as f:
    train_data = f.read()

# Tokenize the data (inputs and labels are the same for causal language modeling)
inputs = tokenizer(train_data, return_tensors="pt", max_length=512, truncation=True)
inputs['labels'] = inputs.input_ids.clone()  # Set labels as a copy of input_ids

# Prepare dataset (inputs and labels)
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.input_ids = inputs['input_ids']
        self.labels = inputs['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'labels': self.labels[idx]}

dataset = ChatDataset(inputs)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

response_ids = model.generate(
    input_ids,
    max_new_tokens=100  # Set this to a reasonable number for your responses
)
