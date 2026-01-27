import textworld
from gymnasium.vector import SyncVectorEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import time

from src.hyperparam import *
from src.env import *
from model.output_func import *

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
    device_map=device,
    trust_remote_code=True
)
model.load_state_dict(torch.load(sft_model_pth)) # finetuned
model.eval()
print(f'{model.device = }')


with open(test_txt_name, 'a', encoding = 'utf-8') as f:
    f.write(f'{inference_type = }\n')

prompt, _ = envs.reset()
start_time = time.perf_counter()
for epoch in range(test_epoch_num):
    epoch_start_time = time.perf_counter()
    for i in tqdm(range(test_epoch_steps)):
        if inference_type == 'ReAct' or inference_type =='ReAct-Im':
            prompt = [env.add_think(t) for env, t in zip(envs.envs, inputprompt_to_outputprompt(model, tokenizer, prompt, '>', think_new_token))]
        prompt, reward, terminated, truncated, infos = envs.step(inputprompt_to_outputprompt(model, tokenizer, prompt, '\n', action_new_token))
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