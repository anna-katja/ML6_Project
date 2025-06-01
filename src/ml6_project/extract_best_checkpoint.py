import json
import os
from glob import glob

trial_path = "../../finetuned/bart_output_large/trial-1"
checkpoint_states = glob(os.path.join(trial_path, "checkpoint-*/trainer_state.json"))

best = None

for state_path in checkpoint_states:
    try:
        with open(state_path) as f:
            data = json.load(f)
            best_ckpt = data.get("best_model_checkpoint", "")
            best_metric = data.get("best_metric", 0.0)

            if best_ckpt and (best is None or best_metric > best["metric"]):
                step = int(best_ckpt.split("-")[-1])
                for log in data.get("log_history", []):
                    if log.get("step") == step:
                        best = {
                            "checkpoint": best_ckpt,
                            "metric": best_metric,
                            "epoch": log.get("epoch"),
                            "step": step
                        }
                        break
    except Exception as e:
        print(f"Failed to process {state_path}: {e}")

if best:
    print(" Best checkpoint found:")
    print(f" Path: {best['checkpoint']}")
    print(f" Step: {best['step']}")
    print(f" Epoch: {best['epoch']}")
    print(f" ROUGE-2: {best['metric']}")
else:
    print(" No best checkpoint found.")

from transformers import AutoTokenizer, AutoModelForSeq2Seq

if best:
    print("\n Loading best model from checkpoint...")
    model_path = best["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(" Model and tokenizer loaded.")
else:
    print(" Cannot load model — no best checkpoint found.")

# to save the model in best model directory cp -r ./bart_output_small/trial-11/checkpoint-1752 ./bart_output_small/best_model_small