import textworld
from tqdm import tqdm
import random
import json

from src.hyperparam import *

def normalize_newlines(s):
    return s.replace("\n", " ") + '\n'

def generate_new_game(options, seeds):
    options.seeds = seeds
    game_file, _ = textworld.make(options)
    env = textworld.start(game_file)
    env.reset()
    return env

# env
options = textworld.GameOptions()
options.nb_objects = nb_objects
options.quest_length = quest_length
seed = 1
env = generate_new_game(options, seed)

pres_step = 0
prompt = env.state.feedback[1210:]
for i in tqdm(range(get_data_num)):
    proper_action = env.state['extra.walkthrough'][pres_step]
    # add train data (prompt, properaction)
    # Q = prompt + f"Possible Actions: {', '.join(env.state.possible_admissible_commands)}\nAction: "
    # A = proper_action + '\n'
    # 저장
    with open("./train_data/textworld_sft.jsonl", "a") as f:
        sample = {
            "prompt": prompt + f"Possible Actions: {', '.join(env.state.possible_admissible_commands)}\nAction: ",
            "action": proper_action + '\n',
        }
        f.write(json.dumps(sample) + "\n")

    env.step(proper_action)
    prompt += f"> {proper_action}\n{normalize_newlines(env.state.feedback)}"
    pres_step += 1
    if env.state.won:
        seed += 1
        env = generate_new_game(options, seed)
        pres_step = 0
        prompt = env.state.feedback[1210:]
        pres_step
