import os
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Path to the directory where cached models are stored
CACHE_DIR = "./model_cache"
DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-ga-en"

def fine_tune_model(feedback_file_path):
    """
    Fine-tune the MarianMT model on new feedback data and overwrite the cached model.

    Args:
        feedback_file_path (str): Path to the feedback file containing training data.
    """
    # Load the feedback data
    if not os.path.exists(feedback_file_path):
        print(f"Feedback file '{feedback_file_path}' does not exist.")
        return

    # Prepare the fine-tuning dataset
    dataset = load_dataset("text", data_files={"train": feedback_file_path})
    dataset = dataset["train"].train_test_split(test_size=0.1)

    # Load the default model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(
        CACHE_DIR if os.path.exists(os.path.join(CACHE_DIR, "config.json")) else DEFAULT_MODEL_NAME
    )
    model = MarianMTModel.from_pretrained(
        CACHE_DIR if os.path.exists(os.path.join(CACHE_DIR, "config.json")) else DEFAULT_MODEL_NAME
    )

    def preprocess_function(examples):
        """
        Preprocess the input text for fine-tuning.

        Args:
            examples (dict): Dictionary of examples with "text" as the key.

        Returns:
            dict: Processed input for the model.
        """
        inputs = ["translate Irish to English: " + example for example in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        return model_inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        save_steps=10,
        logging_dir="./logs",
        predict_with_generate=True,
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Overwrite the cached model with the fine-tuned model
    os.makedirs(CACHE_DIR, exist_ok=True)
    model.save_pretrained(CACHE_DIR)
    tokenizer.save_pretrained(CACHE_DIR)
    print(f"Model has been fine-tuned and saved to {CACHE_DIR}")