from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizer


def load_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = load_dataset(cfg["data"]["dataset"])
    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["checkpoint"])
    max_length = cfg["data"]["max_length"]

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_val = dataset["train"].train_test_split(
        test_size=cfg["data"]["val_split"],
        seed=cfg["data"]["seed"],
    )
    splits = {
        "train": train_val["train"],
        "val": train_val["test"],
        "test": dataset["test"],
    }

    cols = ["input_ids", "attention_mask", "label"]
    loaders = {}
    for name, split in splits.items():
        split = split.map(tokenize, batched=True)
        split.set_format(type="torch", columns=cols)
        loaders[name] = DataLoader(
            split,
            batch_size=cfg["training"]["batch_size"],
            shuffle=(name == "train"),
        )

    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    return loaders["train"], loaders["val"], loaders["test"]
