import argparse
import preprocessing
from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import optuna
import os
import torch

# dataset preprocessing
def preprocess(tokenizer, dataset):
  prefix = "summarize: "
  pre_data = preprocessing.minimal_preprocessing(dataset)
  inputs = [prefix + doc for doc in pre_data["article"]]
  model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
  labels = tokenizer(text_target=dataset["highlights"], max_length=128, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

# compute metrics function

def compute_metrics(tokenizer, rouge):
    def _compute(eval_pred):
      predictions, labels = eval_pred
      decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
      decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    torch.cuda.set_per_process_memory_fraction(0.8, device=0)

    parser = argparse.ArgumentParser(description="T5 fine-tuning")
    parser.add_argument("--model_name", default="t5-small")
    parser.add_argument("--output_dir", default="./t5_best")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rouge = evaluate.load("rouge")

    print("loading data")
    full_data = load_dataset("abisee/cnn_dailymail", "3.0.0")
    # create a smaller dataset for now
    data = preprocessing.custom_dataset_size(full_data, 10000)
    # check if everything looks good with the dataset
    print("Small dataset:", data)
    tokenized_data = data.map(preprocess, tokenizer)
    print("Tokenized dataset:", tokenized_data)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_name)

    # optuna objective
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rage", 1e-5, 5e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
        train_batch_size = trial.suggest_categorial("train_batch_size", [8, 16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)

        arguments = Seq2SeqTrainingArguments (
            output_dir = os.path.join(args.output_dir, f"trial--{trial.number}"),
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = train_batch_size,
            logging_steps = 8,
            num_train_epochs = num_train_epochs,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            predict_with_generate = True,
            metric_for_best_model = "rouge2",
            load_best_model_at_end = True,
            report_to = "none",
            seed = 42,
        )

        trainer = Seq2SeqTrainer(
            model_init = lambda: model_init(args.model_name),
            args = arguments,
            train_dataset = tokenized_data["train"],
            eval_dataset = tokenized_data["val"],
            data_collator = data_collator,
            compute_metrics = compute_metrics(tokenizer, rouge),
            #callbacks = [RichProgressCallback],
        )

        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_rouge2"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial: ", study.best_trial.number, study.best_params)
    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

if __name__ == "__main__":
    main()


