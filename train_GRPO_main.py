import textworld
from gymnasium.vector import SyncVectorEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import time

from src.hyperparam import *
from src.env import *
from model.lora import *

options = textworld.GameOptions()
options.nb_objects = nb_objects
options.quest_length = quest_length

# make env
def make_env(options):
    def _init():
        return TextWorldEnvRL(options, train_seed_num, max_steps=max_steps)
    return _init
envs = SyncVectorEnv([
    make_env(options) for _ in range(num_cpu)
])

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
model.load_state_dict(torch.load(sft_model_pth)) # sft finetuned
apply_lora_for_casulLM(model, r=4, alpha=8, device=device)
model.train()

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

for update in tqdm(range(grpo_updates)):
    optimizer.zero_grad()

    prompts, _ = envs.reset()
    prompts = list(prompts)

    batch_loss = 0.0

    losses = []
    for env_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).input_ids.to(model.device)

        with torch.no_grad():
            enable_lora(model)
            gen_outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=action_new_token,
                do_sample=True,
                temperature=0.8,
                top_p=1,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        action_ids = gen_outputs[:, input_ids.shape[1]:]
        action_texts = tokenizer.batch_decode(
            action_ids,
            skip_special_tokens=True
        )
        action_texts = [
            t[1:max(t.rfind('.'), t.rfind('?')) + 1]
            if ('.' in t or '?' in t) else t[1:]
            for t in action_texts
        ]
        action_texts = [t[:t.find('\n')] if '\n' in t else t for t in action_texts]

        rewards = []
        for action in action_texts:
            ########### step like 구현하기 ##########################
            _, reward, done, _ = envs.envs[env_idx].step_like(action)
            rewards.append(reward)

        rewards = torch.tensor(
            rewards,
            dtype=torch.bfloat16,
            device=model.device
        )

        advantages = rewards - rewards.mean()
        advantages = advantages / (rewards.std() + 1e-6)
        advantages = advantages.detach()

        outputs = model(
            input_ids=gen_outputs,
            labels=gen_outputs
        )
        log_probs = -outputs.loss

        with torch.no_grad():
            disable_lora(model)
            ref_outputs = model(
                input_ids=gen_outputs,
                labels=gen_outputs
            )
            ref_log_probs = -ref_outputs.loss

        pg_loss = -(log_probs * advantages.mean())
        kl_loss = kl_coef * (log_probs - ref_log_probs)

        loss = pg_loss + kl_loss
        batch_loss += loss
        losses.append(loss)

    batch_loss.backward()
    optimizer.step()
    tqdm.write(f"Loss: {sum(losses):.4f}")



merge_lora_in_self_attn(model)

print(model)
torch.save(model.state_dict(), grpo_model_pth)