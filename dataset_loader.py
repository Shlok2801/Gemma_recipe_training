import json
from datasets import Dataset
from config import DATA_PATH

def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    dataset = Dataset.from_list(recipes)

    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    return train_dataset, val_dataset


