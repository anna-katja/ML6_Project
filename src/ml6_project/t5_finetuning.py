import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import argparse
import preprocessing
from datasets import load_dataset
import numpy as np
import evaluate
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import optuna
import torch

# dataset preprocessing
def preprocess(dataset, tokenizer):
    prefix = "summarize: "
    article = preprocessing.minimal_preprocessing(dataset["article"])
    #article = dataset["article"]
    input_text = prefix + article
    model_inputs = tokenizer(input_text, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(text_target=dataset["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
    #print("GPU Available:", torch.cuda.is_available())
    #print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    #torch.cuda.set_per_process_memory_fraction(0.8, device=0)

    best_dir = "t5_small_15_trials_best_additional_preprocess"

    parser = argparse.ArgumentParser(description="T5 fine-tuning")
    parser.add_argument("--model_name", default="t5-small")     #t5-small
    parser.add_argument("--output_dir", default=f"./{best_dir}")
    parser.add_argument("--n_trials", type=int, default=15)
    args = parser.parse_args()

    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("loading data")
    full_data = load_dataset("abisee/cnn_dailymail", "3.0.0")
    data = preprocessing.custom_dataset_size(full_data, 10000)
    tokenized_data = data.map(lambda x: preprocess(x, tokenizer), batched=False)
    print(tokenized_data)

    print("loading rouge")
    rouge = evaluate.load("rouge")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_name, padding=True, return_tensors="pt")

    # optuna objective
    best_score = None
    def objective(trial):
        global best_score

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 2, 10)
        train_batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)

        arguments = Seq2SeqTrainingArguments(
            output_dir = os.path.join(args.output_dir, f"trial_{trial.number}"),
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = train_batch_size,
            logging_steps = 50,
            num_train_epochs = num_train_epochs,
            eval_strategy = "epoch",
            save_strategy = "epoch", # "no"
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            predict_with_generate = True,
            metric_for_best_model = "eval_rouge2",
            load_best_model_at_end = True,
            save_total_limit = 1,
            greater_is_better = True,
            report_to = "none",
            fp16 = torch.cuda.is_available(),
            seed = 42,
        )

        trainer = Seq2SeqTrainer(
            model_init = lambda: model_init(args.model_name),
            args = arguments,
            train_dataset = tokenized_data["train"],
            eval_dataset = tokenized_data["val"],
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = compute_metrics(tokenizer, rouge),
        )

        try:
            trainer.train()
            metrics = trainer.evaluate()
            trainer.save_model(os.path.join(args.output_dir, f"trial_{trial.number}"))
            return metrics["eval_rouge2"]
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    print(f"Best Trial: {best_trial.number}")
    print(f"Best parameters: {best_trial.params}")
    print(f"Best Rouge Score: {best_trial.value}")

    os.makedirs(os.path.join(args.output_dir, best_dir), exist_ok=True)
    with open(os.path.join(args.output_dir, best_dir, "best_params.txt"), "w") as f:
        f.write(f"Trial: {best_trial.number}\n")
        f.write(f"Params: {best_trial.params}\n")
        f.write(f"ROUGE-2: {best_trial.value}\n")

if __name__ == "__main__":
    main()



#t5 large: python t5_finetuning.py --model_name t5-large --output_dir ./t5_large_15_trials_best_additional_preprocess --n_trials 15

