# baseline pretrained models load and evaluate
# compare each pretrained model to finetuned version of that model
#TODO: change line 78 test range
#TODO: change paths to models / checkpoints after retraining

import evaluate
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

from preprocessing import custom_dataset_size


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
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=142, num_beams=5)        #NB: .generate() sets model to eval() mode!
        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_ref = example["highlights"]

        predictions.append(decoded_pred)
        references.append(decoded_ref)

    results = {}
    for metric_name, metric in eval_metrics.items():
        if metric_name == "bertscore":
            bert = metric.compute(predictions=predictions, references=references, lang="en")
            results[metric_name] = {
                "precision": round(float(sum(bert["precision"])) / len(bert["precision"]), 4),
                "recall": round(float(sum(bert["recall"])) / len(bert["recall"]), 4),
                "f1": round(float(sum(bert["f1"])) / len(bert["f1"]), 4)
            }
        else:
            results[metric_name] = {k: round(v, 4) for k, v in metric.compute(predictions=predictions, references=references).items()}

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

full_data = load_dataset("abisee/cnn_dailymail", "3.0.0")
data = custom_dataset_size(full_data, 10000, split_ratio=(0.7, 0.15, 0.15))
test_data = data["test"]     #.select(range(100))
# print(test_data[0])
# print(len(test_data))

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]     #ml6_project
pretrained_dir = project_root/"pretrained"
finetuned_dir = project_root/"finetuned"

pairs_to_evaluate = [
    # {
    #     "pretrained": pretrained_dir/"t5-small",
    #     "fine_tuned": finetuned_dir/"t5_small_15_trials_best"/"trial_0"/"checkpoint-2628"
    # },
    # {
    #     "pretrained": pretrained_dir/"t5-small",
    #     "fine_tuned": finetuned_dir/"t5_small_15_trials_best_additional_preprocess"/"trial_11"/"checkpoint-657"
    # },
    # {
    #     "pretrained": pretrained_dir/"t5-small",
    #     "fine_tuned": finetuned_dir/"t5_small_15_trials_best_no_preprocess"/"trial_3"/"checkpoint-330"
    # },
    # {
    #     "pretrained": pretrained_dir/"t5-large",
    #     "fine_tuned": finetuned_dir/"t5_large_10_trials_best"/"trial_0"/"checkpoint-876"
    # },
    # {
    #     "pretrained": pretrained_dir/"bart-base",
    #     "fine_tuned": finetuned_dir/"bart_output_small"/"best_model_small"/"checkpoint-1752"
    # },
    {
        "pretrained": pretrained_dir/"bart-large",
        "fine_tuned": finetuned_dir/"bart_output_large"/"best_model_large"/"checkpoint-2100"
    },
]

for pair in pairs_to_evaluate:
    compare_models(pair["pretrained"], pair["fine_tuned"], test_data)



"""
output: (full(1500))

Comparison for t5-small:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.3488), 'rouge2': np.float64(0.148), 'rougeL': np.float64(0.2385), 'rougeLsum': np.float64(0.298)}
  bertscore: {'precision': 0.8658, 'recall': 0.8561, 'f1': 0.8608}
Fine-tuned Results (1_best):
  rouge: {'rouge1': np.float64(0.382), 'rouge2': np.float64(0.1704), 'rougeL': np.float64(0.2605), 'rougeLsum': np.float64(0.3259)}
  bertscore: {'precision': 0.8702, 'recall': 0.8687, 'f1': 0.8693}
Fine-tuned Results (2_additional):
  rouge: {'rouge1': np.float64(0.3707), 'rouge2': np.float64(0.1636), 'rougeL': np.float64(0.258), 'rougeLsum': np.float64(0.3189)}
  bertscore: {'precision': 0.875, 'recall': 0.8626, 'f1': 0.8686}
Fine-tuned Results (3_no):
  rouge: {'rouge1': np.float64(0.3733), 'rouge2': np.float64(0.164), 'rougeL': np.float64(0.2526), 'rougeLsum': np.float64(0.3177)}
  bertscore: {'precision': 0.8656, 'recall': 0.8669, 'f1': 0.8661}  


Comparison for t5-large:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.3896), 'rouge2': np.float64(0.1792), 'rougeL': np.float64(0.2722), 'rougeLsum': np.float64(0.3352)}
  bertscore: {'precision': 0.8675, 'recall': 0.8709, 'f1': 0.8691}
Fine-tuned Results:
  rouge: {'rouge1': np.float64(0.4151), 'rouge2': np.float64(0.2001), 'rougeL': np.float64(0.29), 'rougeLsum': np.float64(0.3572)}
  bertscore: {'precision': 0.8786, 'recall': 0.8836, 'f1': 0.8809}
  
  
Comparison for bart-base:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.3659), 'rouge2': np.float64(0.1665), 'rougeL': np.float64(0.2308), 'rougeLsum': np.float64(0.3008)}
  bertscore: {'precision': 0.8526, 'recall': 0.8811, 'f1': 0.8665}
Fine-tuned Results:
  rouge: {'rouge1': np.float64(0.3934), 'rouge2': np.float64(0.1782), 'rougeL': np.float64(0.2713), 'rougeLsum': np.float64(0.3643)}
  bertscore: {'precision': 0.8798, 'recall': 0.8837, 'f1': 0.8816}
  
  
Comparison for bart-large:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.3539), 'rouge2': np.float64(0.1549), 'rougeL': np.float64(0.2211), 'rougeLsum': np.float64(0.2877)}
  bertscore: {'precision': 0.8509, 'recall': 0.8789, 'f1': 0.8645}
Fine-tuned Results:
  rouge: {'rouge1': np.float64(0.3952), 'rouge2': np.float64(0.1774), 'rougeL': np.float64(0.267), 'rougeLsum': np.float64(0.36)}
  bertscore: {'precision': 0.875, 'recall': 0.8869, 'f1': 0.8807}      
"""





"""
output: (range(100))

Comparison for t5-small:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.2544), 'rouge2': np.float64(0.0858), 'rougeL': np.float64(0.1875), 'rougeLsum': np.float64(0.213)}
  bertscore: {'precision': 0.8501, 'recall': 0.8565, 'f1': 0.8532}
Fine-tuned Results (1_best):
  rouge: {'rouge1': np.float64(0.2759), 'rouge2': np.float64(0.1015), 'rougeL': np.float64(0.2006), 'rougeLsum': np.float64(0.2345)}
  bertscore: {'precision': 0.849, 'recall': 0.8658, 'f1': 0.8572}
Fine-tuned Results (2_additional):
  rouge: {'rouge1': np.float64(0.2956), 'rouge2': np.float64(0.1084), 'rougeL': np.float64(0.2126), 'rougeLsum': np.float64(0.2497)}
  bertscore: {'precision': 0.8528, 'recall': 0.8661, 'f1': 0.8593}
Fine-tuned Results (3_no):
  rouge: {'rouge1': np.float64(0.2802), 'rouge2': np.float64(0.1076), 'rougeL': np.float64(0.2009), 'rougeLsum': np.float64(0.2362)}
  bertscore: {'precision': 0.8462, 'recall': 0.8664, 'f1': 0.8561}


Comparison for t5-large:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.2392), 'rouge2': np.float64(0.079), 'rougeL': np.float64(0.173), 'rougeLsum': np.float64(0.205)}
  bertscore: {'precision': 0.8347, 'recall': 0.8562, 'f1': 0.8451}
Fine-tuned Results:
  rouge: {'rouge1': np.float64(0.2639), 'rouge2': np.float64(0.0987), 'rougeL': np.float64(0.1854), 'rougeLsum': np.float64(0.2197)}
  bertscore: {'precision': 0.8453, 'recall': 0.8697, 'f1': 0.8572}
  

Comparison for bart-base:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.2671), 'rouge2': np.float64(0.1199), 'rougeL': np.float64(0.1904), 'rougeLsum': np.float64(0.2199)}
  bertscore: {'precision': 0.8343, 'recall': 0.8805, 'f1': 0.8567}
Fine-tuned Results:
  rouge: {'rouge1': np.float64(0.3383), 'rouge2': np.float64(0.154), 'rougeL': np.float64(0.2518), 'rougeLsum': np.float64(0.3143)}
  bertscore: {'precision': 0.8685, 'recall': 0.8878, 'f1': 0.8779}
  

Comparison for bart-large:
Pretrained Results:
  rouge: {'rouge1': np.float64(0.2605), 'rouge2': np.float64(0.1133), 'rougeL': np.float64(0.1838), 'rougeLsum': np.float64(0.2127)}
  bertscore: {'precision': 0.8332, 'recall': 0.8806, 'f1': 0.8562}
Fine-tuned Results:
  rouge: {'rouge1': np.float64(0.3208), 'rouge2': np.float64(0.135), 'rougeL': np.float64(0.2298), 'rougeLsum': np.float64(0.2909)}
  bertscore: {'precision': 0.8598, 'recall': 0.888, 'f1': 0.8735}
"""
