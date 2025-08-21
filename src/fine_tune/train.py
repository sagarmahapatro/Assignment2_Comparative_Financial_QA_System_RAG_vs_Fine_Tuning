from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from .data  import load_qa_data_for_finetune

MODEL_SIZES = {"small": "gpt2", "medium": "gpt2-medium"}

def fine_tune_model(qa_path, output_dir, model_size, lr, epochs, batch_size):
    print(f"Fine-tuning model with the following parameters:")
    print(f"Q&A Path: {qa_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Size: {model_size}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")

    model_name = MODEL_SIZES.get(model_size, "gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # Fix: set pad_token to eos_token if not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # If we added a new token (optional case), resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dset = load_qa_data_for_finetune(qa_path)
    tokenized = dset.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=512), batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
