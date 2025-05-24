import argparse
import preprocessing
from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
import optuna
import os
import torch


# dataset preprocessing for BART
def preprocess(dataset, tokenizer):
    # BART doesn't need a prefix like T5
    article = preprocessing.minimal_preprocessing(dataset["article"])

    # Tokenize inputs - BART can handle longer sequences well
    model_inputs = tokenizer(
        article,
        max_length=1024,
        truncation=True,
        padding=False  # Don't pad during preprocessing, let DataCollator handle it
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            dataset["highlights"],
            max_length=142,  # BART paper uses 142 for CNN/DailyMail
            truncation=True,
            padding=False
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# compute metrics function
def compute_metrics(tokenizer, rouge):
    def _compute(eval_pred):
        predictions, labels = eval_pred

        # Clean predictions: clip values to valid token range and convert to int
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        predictions = predictions.astype(np.int32)

        # Handle invalid tokens in predictions
        try:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        except (OverflowError, ValueError) as e:
            print(f"Warning: Error decoding predictions: {e}")
            # Fallback: create empty strings for failed predictions
            decoded_preds = [""] * len(predictions)

        # Replace -100 tokens with pad token for decoding labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
        labels = labels.astype(np.int32)

        try:
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except (OverflowError, ValueError) as e:
            print(f"Warning: Error decoding labels: {e}")
            decoded_labels = [""] * len(labels)

        # Ensure we have valid strings for ROUGE computation
        decoded_preds = [pred if pred.strip() else "empty" for pred in decoded_preds]
        decoded_labels = [label if label.strip() else "empty" for label in decoded_labels]

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    return _compute


# for Optuna hyperparameter optimization
def model_init(model_name):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


# main
def main():
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using device:", torch.cuda.get_device_name(0))
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
    else:
        print("Using device: CPU")

    parser = argparse.ArgumentParser(description="BART fine-tuning")
    parser.add_argument("--model_name", default="facebook/bart-base")
    parser.add_argument("--output_dir", default="./bart_base_")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Ensure tokenizer has pad token (some BART tokenizers don't by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token #


    print("Loading data")
    full_data = load_dataset("abisee/cnn_dailymail", "3.0.0")
    # Create a smaller dataset for now
    data = preprocessing.custom_dataset_size(full_data, 10000)
    # Check if everything looks good with the dataset
    print("Small dataset:", data)

    tokenized_data = data.map(lambda x: preprocess(x, tokenizer), batched=False)
    print("Tokenized dataset:", tokenized_data)

    print("Loading rouge")
    rouge = evaluate.load("rouge")

    # BART-specific data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=args.model_name,
        padding=True,
        return_tensors="pt"
    )

    # Optuna objective function
    def objective(trial):
        # BART-specific hyperparameter ranges based on literature
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)
        train_batch_size = trial.suggest_categorical("train_batch_size", [4, 8, 16])  # Smaller for BART
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
        warmup_steps = trial.suggest_int("warmup_steps", 500, 2000)

        arguments = Seq2SeqTrainingArguments(
            output_dir=os.path.join(args.output_dir, f"trial--{trial.number}"),
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            gradient_accumulation_steps=1,
            logging_steps=50,
            num_train_epochs=num_train_epochs,
            eval_strategy="epoch",  # Some versions need this instead of eval_strategy
            save_strategy="epoch",
            save_total_limit=1,  # Only keep the best checkpoint
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            predict_with_generate=True,
            metric_for_best_model="eval_rouge2",
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to="none",
            seed=42,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_pin_memory=False,
            # Generation parameters for evaluation
            generation_max_length=142,  # Standard for CNN/DailyMail
            generation_min_length=56,  # Minimum summary length
            generation_num_beams=4,
            generation_length_penalty=2.0,
            generation_early_stopping=True,
        )

        trainer = Seq2SeqTrainer(
            model_init=lambda: model_init(args.model_name),
            args=arguments,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["val"],
            data_collator=data_collator,
            compute_metrics=compute_metrics(tokenizer, rouge),
        )

        try:
            trainer.train()
            trainer.save_model(os.path.join(args.output_dir, f"trial--{trial.number}"))
            metrics = trainer.evaluate()
            return metrics["eval_rouge2"]
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a low score instead of crashing
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print(f"BART best trial: {study.best_trial.number}, {study.best_params}")
    best_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    # Save the best hyperparameters
    with open(os.path.join(best_dir, "best_params.txt"), "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Best ROUGE-2 score: {study.best_trial.value}\n")


if __name__ == "__main__":
    main()