# Configure poetry venv and interpreter
path to poetry env: /home/mlt_ml6/.cache/pypoetry/virtualenvs/ml6-project-Bz2GhCbn-py3.11

for interpreter (Pycharm): add new > local > select existing > python > /home/mlt_ml6/.cache/pypoetry/virtualenvs/ml6-project-Bz2GhCbn-py3.11/bin/python

for terminal (make sure correct working directory is configured): source $(poetry env info --path)/bin/activate

cd mlt_ml6@ai-server:/mnt/nvme/home/mlt_ml6/ml6_project


# Dataset
CNN/DailyMail dataset available at: https://huggingface.co/datasets/abisee/cnn_dailymail (3.0.0)


# Models
- BART-large
- BART-base
- T5-large
- T5-small
