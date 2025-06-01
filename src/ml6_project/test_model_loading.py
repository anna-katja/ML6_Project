from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def test_model(path: Path):
    print(f"\nTesting: {path}")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model failed to load: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Tokenizer failed to load: {e}")


this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

pretrained_dir = project_root/"pretrained"
finetuned_dir = project_root/"finetuned"

model_paths = [
    ("Pretrained t5-small", pretrained_dir/"t5-small"),
    ("Fine-tuned t5-small", finetuned_dir/"t5_small_15_trials_best"/"trial_0"/"checkpoint-2628"),
    ("Fine-tuned t5-small (additional preprocessing)", finetuned_dir/"t5_small_15_trials_best_additional_preprocess"/"trial_11"/"checkpoint-657"),
    ("Fine-tuned t5-small (no preprocessing)", finetuned_dir/"t5_small_15_trials_best_no_preprocess"/"trial_3"/"checkpoint-330"),

    ("Pretrained t5-large", pretrained_dir/"t5-large"),
    ("Fine-tuned t5-large", finetuned_dir/"t5_large_10_trials_best"/"trial_0"/"checkpoint-876"),

    ("Pretrained bart-base", pretrained_dir/"bart-base"),
    ("Fine-tuned bart-base", finetuned_dir/"bart_output_small"/ "best_model_small"/"checkpoint-1752"),

    ("Pretrained bart-large", pretrained_dir/"bart-large"),
    ("Fine-tuned bart-large", finetuned_dir/"bart_output_large"/"best_model_large"/"checkpoint-2100"),
]

for label, path in model_paths:
    print(f"\n=== {label} ===")
    test_model(path)
