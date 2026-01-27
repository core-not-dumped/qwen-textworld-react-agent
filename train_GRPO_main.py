import textworld
from gymnasium.vector import SyncVectorEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
from collections import deque

from src.hyperparam import *
from src.env import *
from model.lora import *
from rl.loss import *

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

# newlines까지 학습시킬것임
newline_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
prompts, _ = envs.reset()

# 학습 루프
batch_losses = deque(maxlen=50)
for update in tqdm(range(grpo_updates)):
    optimizer.zero_grad()

    prompts = list(prompts)

    batch_loss = 0.0

    for env_idx, prompt in enumerate(prompts):
        input_id = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).input_ids.to(model.device)
        
        with torch.no_grad():
            enable_lora(model)
            gen_outputs = model.generate(
                input_ids=input_id,
                max_new_tokens=action_new_token,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        action_ids = gen_outputs[:, input_id.shape[1]:]

        # \n이 처음 나온 position을 first_pos에 저장
        mask = (action_ids == newline_id)
        first_pos = torch.full((group_size,), action_new_token, device=action_ids.device)
        has_newline = mask.any(dim=1)
        first_pos[has_newline] = mask[has_newline].float().argmax(dim=1)

        # action_texts를 str형태로 저장 (\n포함안됨)
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

        # 각 action의 reward계산
        rewards = []
        for action in action_texts:
            reward = envs.envs[env_idx].step_reward(action)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.bfloat16, device=model.device)

        # advantage계산
        advantages = rewards - rewards.mean()
        advantages = advantages.detach()

        # log prob 계산
        outputs = model(input_ids=gen_outputs, labels=gen_outputs)
        action_mask = torch.zeros(gen_outputs.shape, device=gen_outputs.device)
        for i in range(group_size): action_mask[i, input_id.shape[1]:input_id.shape[1]+first_pos[i]+1] = 1.0
        log_probs = compute_logprob(outputs.logits, gen_outputs, action_mask)

        # reg log prob 계산
        with torch.no_grad():
            disable_lora(model)
            ref_outputs = model(input_ids=gen_outputs, labels=gen_outputs)
            ref_log_probs = compute_logprob(ref_outputs.logits, gen_outputs, action_mask)

        # total loss계산
        pg_loss = -(log_probs * advantages).mean()
        kl_loss = kl_coef * (log_probs - ref_log_probs).mean()
        loss = pg_loss + kl_loss
        
        batch_loss += loss

        reward_max_idx = rewards.argmax().item()
        prompts[env_idx], _, terminated, truncated, _ = envs.envs[env_idx].step(action_texts[reward_max_idx])
        if terminated or truncated: prompts[env_idx], _ = envs.envs[env_idx].reset()
    
    batch_loss = batch_loss / num_cpu
    batch_loss.backward()
    optimizer.step()
    batch_losses.append(batch_loss.item())
    tqdm.write(f"Loss: {sum(batch_losses)/len(batch_losses):.4f}")

merge_lora_in_self_attn(model)

print(model)
torch.save(model.state_dict(), grpo_model_pth)


# sft action mask만들기