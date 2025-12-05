import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configuration Constants
MODEL_ID = "BioMistral/BioMistral-7B"
# Replace with your local dataset path (e.g., 'your_rag_data.csv')
DATASET_NAME = "rag_instruction_dataset.csv"
OUTPUT_DIR = "./llama3_qlora_rag_adapter"

# Log less verbosely
logging.set_verbosity_warning()

# 2. QLoRA (4-bit Quantization) Configuration
# This is the core of the memory reduction technique
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat 4-bit (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
    bnb_4bit_use_double_quant=True
)

# 3. Load Model and Tokenizer
print(f"Loading Model: {MODEL_ID} with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False # Disable cache for training
model = prepare_model_for_kbit_training(model) # Prepare for 4-bit training

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for causal models

# 4. LoRA Configuration
# This defines the small matrices (adapters) that will be trained
lora_config = LoraConfig(
    r=16, # Rank: Controls the size of the adapter (smaller = faster, less expressive)
    lora_alpha=16, # Scaling factor for the weights
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Target attention matrices (q_proj and v_proj are standard)
    target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
)

# 5. Load and Format Dataset
# IMPORTANT: You need to create this dataset from your RAG data (question/answer pairs)
dataset = load_dataset("csv", data_files=DATASET_NAME, split="train")

def formatting_func(example):
    # This formats your data into the instruction-following structure Llama 3 expects
    output_text = f"Question: {example['question']}\n\nAnswer: {example['answer']}{tokenizer.eos_token}"
    return {"text": output_text}

dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)

# 6. Training Arguments
# Use minimal settings for a quick run
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit", # Optimize memory for AdamW
    learning_rate=2e-4,
    fp16=False, # Handled by bfloat16 in BitsAndBytesConfig
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 7. SFT Trainer and Training Start
print("Starting Supervised Fine-Tuning (SFT) with QLoRA...")
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512, # Max length of input sequence
)

trainer.train()

# 8. Save the LoRA Adapter Weights (the small, trained part)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ QLoRA adapter saved to: {OUTPUT_DIR}")