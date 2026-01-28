import textworld
from tqdm import tqdm
from transformers import AutoTokenizer
import heapq

from src.hyperparam import *

def normalize_newlines(s):
    return s.replace("\n", " ") + '\n'

def generate_new_game(options, seeds):
    options.seeds = seeds
    game_file, _ = textworld.make(options)
    env = textworld.start(game_file)
    env.reset()
    return env

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"
)

# env
options = textworld.GameOptions()
options.nb_objects = nb_objects
options.quest_length = quest_length
seed = 1
env = generate_new_game(options, seed)

max_action_len = []
top_n = 100
pres_step = 0
prompt = env.state.feedback[1210:]
for i in tqdm(range(get_data_num)):
    proper_action = env.state['extra.walkthrough'][pres_step]
    tokens = tokenizer(proper_action, add_special_tokens=False).input_ids
    if len(max_action_len) < top_n: heapq.heappush(max_action_len, len(tokens)+1)  # 아직 top_n 안 찼으면 그냥 넣기
    else:   heapq.heappushpop(max_action_len, len(tokens)+1)  # top_n보다 크면 교체
    if i % 100 == 0: print(f'{max_action_len = }')

    env.step(proper_action)
    prompt += f"> {proper_action}\n{normalize_newlines(env.state.feedback)}"
    pres_step += 1
    if env.state.won:
        seed += 1
        env = generate_new_game(options, seed)
        pres_step = 0
        prompt = env.state.feedback[1210:]
        pres_step
