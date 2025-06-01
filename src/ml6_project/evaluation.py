# baseline pretrained models load and evaluate
# compare each pretrained model (load from hf) to finetuned version of that model (local)
#TODO: change line 71 test range

import evaluate
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path


def evaluate_model(model_path, test_dataset, metric_names=["rouge", "bertscore"]):
    model_path = Path(model_path).resolve()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded model from {model_path} on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_metrics = {metric: evaluate.load(metric) for metric in metric_names}

    predictions = []
    references = []

    for example in test_dataset:
        inputs = tokenizer(example["article"], max_length=1024, truncation=True, padding=True, return_tensors="pt")
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=142, num_beams=5)
        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_ref = example["highlights"]

        predictions.append(decoded_pred)
        references.append(decoded_ref)

    results = {}
    for metric_name, metric in eval_metrics.items():
        if metric_name == "bertscore":
            results[metric_name] = metric.compute(predictions=predictions, references=references, lang="en")
        else:
            results[metric_name] = metric.compute(predictions=predictions, references=references)

    return results


def compare_models(pretrained_model_path, fine_tuned_model_path, test_dataset):
    print(f"Evaluating {pretrained_model_path.name} (Pretrained)...")
    pretrained_results = evaluate_model(pretrained_model_path, test_dataset)

    print(f"Evaluating {pretrained_model_path.name} (Fine-tuned)...")
    fine_tuned_results = evaluate_model(fine_tuned_model_path, test_dataset)

    print(f"Comparison for {pretrained_model_path.name}:")
    print("Pretrained Results:")
    for metric, scores in pretrained_results.items():
        print(f"  {metric}:", {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})
    print("Fine-tuned Results:")
    for metric, scores in fine_tuned_results.items():
        print(f"  {metric}:", {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})
    print()


dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
test_data = dataset["test"].select(range(100))        #test
# print(test_data[0])

this_file = Path(__file__).resolve()
project_root = this_file.parents[1]     #ml6_project

pretrained_dir = project_root/"pretrained"
finetuned_dir = project_root/"finetuned"

pairs_to_evaluate = [
    {
        "pretrained": pretrained_dir/"t5-small",
        "fine_tuned": finetuned_dir/"t5_small_15_trials_best_additional_preprocess"/"trial_0"
    },
    {
        "pretrained": pretrained_dir/"t5-large",
        "fine_tuned": finetuned_dir/"t5_large_10_trials_best"/"trial_0"
    },
    {
        "pretrained": pretrained_dir/"bart-base",
        "fine_tuned": finetuned_dir/"bart_output_small"/"best_model_small"
    },
    {
        "pretrained": pretrained_dir/"bart-large",
        "fine_tuned": finetuned_dir/"bart_output_large"/"best_model_large"
    },
]

for pair in pairs_to_evaluate:
    compare_models(pair["pretrained"], pair["fine_tuned"], test_data)





"""
output:



"""