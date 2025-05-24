import os
import re
import numpy as np
import contractions
import evaluate
import optuna
import torch

from datasets import load_dataset, DatasetDict
from nltk.corpus import stopwords
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)

# Constants
MODEL_NAME = "facebook/bart-base"
OUTPUT_DIR = "./bart_base_optuna"

# Dataset handling
def custom_dataset_size(dataset, size, split_ratio=(0.7, 0.15, 0.15)):
    assert isinstance(dataset, DatasetDict)
    train_size = round(size * split_ratio[0])
    val_size = round(size * split_ratio[1])
    test_size = round(size * split_ratio[2])

    shuffled = dataset["train"].shuffle(seed=42)
    return DatasetDict({
        "train": shuffled.select(range(train_size)),
        "val": shuffled.select(range(train_size, train_size + val_size)),
        "test": dataset["test"].shuffle(seed=42).select(range(test_size))
    })

# Text cleaning
metadata_patterns = [
    r"^[^(]*\([^\)]*\)\s*--\s*",
    r"^.*UPDATED:\s+\.\s+\d{2}:\d{2}\s+\w+,\s+\d+\s+\w+\s+\d{4}\s+\.\s+",
    r"^By\s+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\.\s+(?:and\s+Associated\s+Press\s+Reporter\s*\.\s*)?",
    r"^(\([^\)]*\))",
    r"^By\s+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\.\s+PUBLISHED:\s+\.\s+\d{2}:\d{2}\s+EST,\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\.\s+\|\s+\.\s+UPDATED:\s+\.\s+\d{2}:\d{2}\s+EST,\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\.\s+"
]

def minimal_preprocessing(article):
    for pattern in metadata_patterns:
        article = re.sub(pattern, '', article)
    return contractions.fix(article).strip()

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocessing

def preprocess(example):
    article = minimal_preprocessing(example["article"])
    model_inputs = tokenizer(article, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["highlights"], max_length=142, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() if pred.strip() else "empty" for pred in decoded_preds]
    decoded_labels = [label.strip() if label.strip() else "empty" for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

# Model init
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load and process data
dataset = load_dataset("cnn_dailymail", "3.0.0")
small_dataset = custom_dataset_size(dataset, size=10000)
tokenized_data = small_dataset.map(preprocess, batched=False)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)

# Optuna objective
def objective(trial):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{OUTPUT_DIR}/trial-{trial.number}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical("train_batch_size", [4, 8, 16]),
        per_device_eval_batch_size=8,
        weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1),
        num_train_epochs=trial.suggest_int("num_train_epochs", 3, 6),
        warmup_steps=trial.suggest_int("warmup_steps", 500, 2000),
        logging_steps=50,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge2",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42
    )

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    try:
        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_rouge2"]
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

# Run study
if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Params: {study.best_params}")
