# baseline pretrained models load and evaluate
# compare each pretrained model (load from hf) to finetuned version of that model (local)
#TODO: change line 71 test range

import evaluate
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path


def evaluate_model(model_name, test_dataset, metric_names=["rouge", "bertscore"], fine_tuned_model_path=None):
    model_path = fine_tuned_model_path if fine_tuned_model_path else model_name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded model on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_metrics = {metric: evaluate.load(metric) for metric in metric_names}

    predictions = []
    references = []

    # generate predictions for the test dataset
    for example in test_dataset:
        inputs = tokenizer(example["article"], max_length=1024, truncation=True, padding=True, return_tensors="pt")
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=142, num_beams=5)
        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_ref = example["highlights"]

        predictions.append(decoded_pred)
        references.append(decoded_ref)

    # compute metrics
    results = {}
    for metric_name, metric in eval_metrics.items():
        if metric_name == "bertscore":
            results[metric_name] = metric.compute(predictions=predictions, references=references, lang="en")    #bertscore
        else:
            results[metric_name] = metric.compute(predictions=predictions, references=references)       #rouge

    return results


def compare_models(pretrained_model_name, fine_tuned_model_path, test_dataset):
    print(f"Evaluating {pretrained_model_name} (Pretrained)...")
    pretrained_results = evaluate_model(pretrained_model_name, test_dataset)

    print(f"Evaluating {pretrained_model_name} (Fine-tuned)...")
    fine_tuned_results = evaluate_model(pretrained_model_name, test_dataset, fine_tuned_model_path=fine_tuned_model_path)

    print(f"Comparison for {pretrained_model_name}:")
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

t5_small_path = project_root/"t5_small_15_trials_best"/"trial_0"/"checkpoint-1752"
t5_large_path = project_root/"t5_large_10_trials_best"/"trial_0"/"checkpoint-876"
bart_base_path = project_root/"bart_output_small"/"best_model_small"/"checkpoint-1752"
bart_large_path = project_root/"bart_output_large"/"best_model_large"/"checkpoint-2100"


pairs_to_evaluate = [
    {"pretrained": "t5-small", "fine_tuned": t5_small_path},
    {"pretrained": "t5-large", "fine_tuned": t5_large_path},
    {"pretrained": "facebook/bart-base", "fine_tuned": bart_base_path},
    {"pretrained": "facebook/bart-large", "fine_tuned": bart_large_path}
]

for pair in pairs_to_evaluate:
    compare_models(pair["pretrained"], pair["fine_tuned"], test_data)





"""
output:



"""