import textworld
from gymnasium.vector import SyncVectorEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import time

from src.hyperparam import *
from src.env import *

options = textworld.GameOptions()
options.nb_objects = nb_objects
options.quest_length = quest_length

# make env
def make_env(options):
    def _init():
        return TextWorldEnv(options, inference_type, train_seed_num, max_steps=max_steps)
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
    device_map="auto",
    trust_remote_code=True
)
def inputprompt_to_outputprompt(prompt, split_str = None, max_new_token = 30):
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
    print(inputs)

    # implementation
    '''
    with torch.no_grad():
        outputs = model.forward(
            **inputs,
            use_cache = True,
        )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    tokens = [next_token]
    with torch.no_grad():
        for _ in range(max_new_token-1):
            outputs = model.forward(
                next_token,
                use_cache = True,
                past_key_values = past_key_values,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            tokens.append(next_token)

    final_outputs = torch.cat(tokens, dim=1)
    output_text = tokenizer.batch_decode(final_outputs, skip_special_tokens=True)
    '''

    # method
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_token,
        do_sample=True,
        temperature=0.8,
        top_p=1,
        min_p=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.batch_decode(outputs[:,-max_new_token:], skip_special_tokens=True)

    output_text = [
        t[1:max(t.rfind('.'), t.rfind('?')) + 1]
        if ('.' in t or '?' in t) else t[1:]
        for t in output_text
    ]
    if split_str != None:
        output_text = [t[:t.find(split_str)] if split_str in t else t for t in output_text]
    return output_text


with open(test_txt_name, 'a', encoding = 'utf-8') as f:
    f.write(f'{inference_type = }\n')

prompt, _ = envs.reset()
start_time = time.perf_counter()
for epoch in range(epoch_num):
    epoch_start_time = time.perf_counter()
    for i in tqdm(range(epoch_steps)):
        if inference_type == 'ReAct' or inference_type =='ReAct-Im':
            prompt = [env.add_think(t) for env, t in zip(envs.envs, inputprompt_to_outputprompt(prompt, '>', think_new_token))]
        prompt, reward, terminated, truncated, infos = envs.step(inputprompt_to_outputprompt(prompt, '\n', action_new_token))
    epoch_end_time = time.perf_counter()

    success_num = sum(e.success for e in envs.envs)
    fail_num = sum(e.fail for e in envs.envs)
    success_rate = success_num / (success_num + fail_num) * 100
    with open(test_txt_name, 'a', encoding = 'utf-8') as f:
        f.write(f'{epoch = }, {success_rate = }%\n')
        f.write(f'{success_num = }, {fail_num = }\n')
        f.write(f'elapsed time = {epoch_end_time - epoch_start_time}s\n')
        print(f'{epoch = }, {success_rate = }%')
        print(f'{success_num = }, {fail_num = }')
        print(f'elapsed time = {epoch_end_time - epoch_start_time}s')

end_time = time.perf_counter()
with open(test_txt_name, 'a', encoding = 'utf-8') as f:
    f.write(f'total elapsed time = {end_time - start_time}s\n')
    print(f'total elapsed time = {end_time - start_time}s')