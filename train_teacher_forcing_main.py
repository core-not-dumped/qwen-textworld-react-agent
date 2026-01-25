from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm
import torch

from src.dataloader import *
from src.hyperparam import *

# make model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

dataset = load_dataset(
    "json",
    data_files="./train_data/textworld_sft.jsonl",
    split="train"
)

def preprocess(example):
    text = (
        "<|user|>\n"
        + example["prompt"].strip()
        + "\n<|assistant|>\n"
        + example["action"].strip()
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding=False,
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=4
)

train_loader = TeacherForcingDataloader(dataset, batch_size=4)
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to("cuda") for k, v in batch.items()}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tqdm.write(f"Loss: {loss.item():.4f}")

