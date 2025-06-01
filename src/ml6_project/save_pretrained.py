from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]     #ml6_project
base_dir = project_root/"pretrained"
base_dir.mkdir(parents=True, exist_ok=True)

model_names = {
    "t5-small": "t5-small",
    "t5-large": "t5-large",
    "bart-base": "facebook/bart-base",
    "bart-large": "facebook/bart-large"
}

for name, hf_name in model_names.items():
    target_path = base_dir / name
    print(f"Saving {hf_name} to {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)

    tokenizer.save_pretrained(target_path)
    model.save_pretrained(target_path)
