import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_model(data_dir, model_variant, output_dir, num_train_epochs, batch_size):
    # Load tokenizer and model based on the specified variant
    tokenizer = GPT2Tokenizer.from_pretrained(model_variant)
    model = GPT2LMHeadModel.from_pretrained(model_variant)

    # Load dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_dir,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset)

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of the training data")
    parser.add_argument("--model_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], help="GPT-2 model variant")
    parser.add_argument("--output_dir", type=str, default="./models/", help="Output directory to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")

    args = parser.parse_args()
    train_model(args.data_dir, args.model_variant, args.output_dir, args.num_train_epochs, args.batch_size)
