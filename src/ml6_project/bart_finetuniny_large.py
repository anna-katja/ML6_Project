import argparse
import os
##
# Set GPU before any torch/transformers imports
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="facebook/bart-base")
parser.add_argument("--output_dir", default="./bart_output")
parser.add_argument("--n_trials", type=int, default=5)
parser.add_argument("--gpu_id", type=int, default=0)
args, _ = parser.parse_known_args()

# Set CUDA device early
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import numpy as np
import torch
import evaluate
import optuna
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import preprocessing
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import preprocessing


def preprocess(example, tokenizer):
    article = preprocessing.minimal_preprocessing(example["article"])
    model_inputs = tokenizer(
        article,
        max_length=1024,
        truncation=True,
        padding=False
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["highlights"],
            max_length=142,
            truncation=True,
            padding=False
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(tokenizer, rouge):
    def _compute(eval_pred):
        predictions, labels = eval_pred
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)

        try:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            print("Decoding error:", e)
            decoded_preds = [""] * len(predictions)
            decoded_labels = [""] * len(labels)

        decoded_preds = [p if p.strip() else "empty" for p in decoded_preds]
        decoded_labels = [l if l.strip() else "empty" for l in decoded_labels]

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions])
        return {k: round(v, 4) for k, v in result.items()}

    return _compute


def model_init(model_name):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/bart-large")
    parser.add_argument("--output_dir", default="./bart_output")
    parser.add_argument("--n_trials", type=int, default=15) #increase to 10
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    dataset = preprocessing.custom_dataset_size(dataset, 20000)
    tokenized = dataset.map(lambda x: preprocess(x, tokenizer), batched=False)

    rouge = evaluate.load("rouge")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_name, padding=True, return_tensors="pt")

    def objective(trial):
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(args.output_dir, f"trial-{trial.number}"),
            per_device_train_batch_size=trial.suggest_categorical("train_batch_size", [4, 8, 16]),
            per_device_eval_batch_size=8,
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True),
            weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1),
            num_train_epochs=trial.suggest_int("num_train_epochs", 2, 12), # change to 2, 12
            warmup_steps=trial.suggest_int("warmup_steps", 500, 2000),
            eval_strategy="epoch",
            save_strategy="no", # change to false/no
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
            model_init=lambda: model_init(args.model_name),
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics(tokenizer, rouge)
        )

        try:
            trainer.train()
            metrics = trainer.evaluate()
            trainer.save_model(os.path.join(args.output_dir, f"trial-{trial.number}"))
            return metrics["eval_rouge2"]
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    print(f"Best Trial: {best_trial.number}")
    print(f"Best Params: {best_trial.params}")
    print(f"Best ROUGE-2 Score: {best_trial.value}")

    os.makedirs(os.path.join(args.output_dir, "best_model"), exist_ok=True)
    with open(os.path.join(args.output_dir, "best_model", "best_params.txt"), "w") as f:
        f.write(f"Trial: {best_trial.number}\n")
        f.write(f"Params: {best_trial.params}\n")
        f.write(f"ROUGE-2: {best_trial.value}\n")


if __name__ == "__main__":
    main()
