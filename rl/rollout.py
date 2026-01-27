import torch
from model.lora import *

def rollout(model, prompts, tokenizer, max_new_tokens=128):
    model.train()
    enable_lora(model)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return outputs