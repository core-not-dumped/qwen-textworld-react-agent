from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
from torch.optim import AdamW
import torch

from src.dataloader import *
from src.hyperparam import *
from model.lora import *

# make model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

dataset = load_dataset(
    "json",
    data_files=train_data_pth,
    split="train"
)

def preprocess(example):
    text = (
        example["prompt"].strip()
        + example["action"].strip()
        + '\n'
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

# ?%만 학습 데이터로 사용
indices = list(range(len(dataset)))
random.shuffle(indices)
subset_indices = indices[:int(0.05 * len(dataset))]
dataset = dataset.select(subset_indices)

# dataset길이순으로 변경
lengths = [len(x) for x in dataset["input_ids"]]
sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
dataset = dataset.select(sorted_indices)
dataset = dataset.select(range(100, len(dataset)))
lengths = [len(x) for x in dataset["input_ids"]]

# trainloader설정
train_loader = TeacherForcingDataloader(dataset, batch_size=bs, shuffle=False)

# lora적용
apply_lora_for_casulLM(model, r=4, alpha=8, device=device)

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
print(model)

model.train()
for epoch in range(train_epoch_num):
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

merge_lora_in_self_attn(model)

print(model)
torch.save(model.state_dict(), sft_model_pth)