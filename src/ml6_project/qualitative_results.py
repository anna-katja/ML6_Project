import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import torch
from datasets import load_dataset

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]     #ml6_project
pretrained_dir = project_root/"pretrained"
finetuned_dir = project_root/"finetuned"

# Hardcode GPUs 1 and 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

text_for_qualitative_results = (
    "Military air strike kills at least 20 people in northwest Nigeria. Amnesty International calls for an investigation into the ‘reckless’ attack in the violence-hit Zamfara state. A military air strike in northwest Nigeria has killed at least 20 people, according to the military and local residents, prompting calls from human rights groups for an investigation into the attack. The strike occurred over the weekend in Zamfara state, one of the regions worst affected by violence from armed groups, commonly referred to as “bandits”. Nigerian Air Commodore Ehimen Ejodame said the strike followed intelligence that 'a significant number of terrorists were massing and preparing to strike unsuspecting settlements. Further intelligence confirmed that the bandits had killed some farmers and abducted a number of civilians, including women and children,' Ejodame said in a statement, adding that two local vigilantes were killed and two others injured in the crossfire. However, according to residents cited by the AFP news agency, a group of local vigilantes pursuing a gang was mistakenly bombed by a Nigerian military jet. The air force had been called in by villagers who had suffered an attack earlier in the weekend. Locals said an unknown number of people were also wounded in the strike. 'We were hit by double tragedy on Saturday,' said Buhari Dangulbi, a resident of the affected area. 'Dozens of our people and several cows were taken by bandits, and those who trailed the bandits to rescue them were attacked by a fighter jet. It killed 20 of them.' Residents told AFP that the bandits had earlier attacked the villages of Mani and Wabi in Maru district, stealing cattle and abducting several people. In response, vigilantes launched a pursuit to recover the captives and stolen livestock. 'The military aircraft arrived and started firing, killing at least 20 of our people,' Abdullahi Ali, a Mani resident and member of a local hunters’ militia, told the Reuters news agency."
)

model_paths_for_inference = [
    {
        "name": "BART Small",
        "model_path": finetuned_dir / "bart_output_small" / "best_model_small" / "checkpoint-1752"
    },
    {
        "name": "BART Large",
        "model_path": finetuned_dir / "bart_output_large" / "best_model_large" / "checkpoint-2100"
    },
    {
        "name": "T5 Small",
        "model_path": finetuned_dir / "t5_small_15_trials_best" / "trial_0" / "checkpoint-2628"
    },
    {
        "name": "T5 Large",
        "model_path": finetuned_dir / "t5_large_10_trials_best" / "trial_0" / "checkpoint-876"
    }
]

def run_qualitative_inference(model_path: Path, dataset, model_name=""):
    model_path = model_path.resolve()

    # loading model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n🔍 Running inference with {model_name} at {model_path}")
    print(f"📦 Model loaded on: {device}\n")

    for i, example in enumerate(dataset):
        input_text = example["article"]
        reference = example.get("highlights", "[No reference available]")

        inputs = tokenizer(input_text, max_length=1024, truncation=True, padding=True, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=142, num_beams=5)

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"=== Example {i + 1} ===")
        print(f"📰 Input (truncated):\n{input_text[:300]}...\n")
        print(f"🧠 Generated Summary:\n{decoded_output}\n")
        print(f"🎯 Reference Summary:\n{reference}")
        print("=" * 80)



if __name__ == "__main__":

    if text_for_qualitative_results.strip():

        dataset = [{"article": text_for_qualitative_results, "highlights": ""}]
    else:
        dataset = load_dataset("cnn_dailymail", "3.0.0")["test"].select(range(1))

    # running inference for each model
    for model_info in model_paths_for_inference:
        run_qualitative_inference(model_info["model_path"], dataset, model_name=model_info["name"])


run_qualitative_inference(
    model_path=model_paths_for_inference[0]["model_path"],
    dataset=dataset,
    model_name=model_paths_for_inference[0]["name"]
)
