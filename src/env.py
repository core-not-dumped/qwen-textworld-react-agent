import gymnasium as gym
from gymnasium import spaces
import textworld
import string
import random

# state dict key
'''
'last_command', 'raw', 'feedback', 'score', 'done', 'game',
'command_templates', 'verbs', 'entities', 'typed_entities', 'possible_commands', 'possible_admissible_commands',
'objective', 'max_score', 'extra.walkthrough', 'extra.uuid', 'moves', 'won', 'lost'
'''

TEXTWORLD_CHARSET = (
    string.ascii_letters +
    string.digits +
    string.punctuation +
    " \n"
)

class TextWorldEnv(gym.Env):
    def __init__(self, options, inference_type, train_seed_num, max_steps=50):
        self.options = options # random.randint(1, 1000)
        self.max_steps = max_steps
        self.train_seed_num = train_seed_num

        self.observation_space = spaces.Text(
            max_length=4096,
            charset=TEXTWORLD_CHARSET
        )
        self.action_space = spaces.Text(
            max_length=256,
            charset=TEXTWORLD_CHARSET
        )

        self.steps = 0
        self.context = ""

        self.success = 0
        self.fail = 0
        self.eps = 1e-5

        self.inference_type = inference_type

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.generate_new_game(self.options, random.randint(self.train_seed_num, self.train_seed_num*10))
        state = self.env.state
        self.context = state.feedback[1210:] # remove TEXT WORLD

        if self.inference_type == 'ReAct':
            obs = self.context + "Think: "
        elif self.inference_type == 'ReAct-Im':
            obs = self.context + "Think: "
        elif self.inference_type == 'Act':
            obs = self.context + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "
        
        return obs, {}

    def step(self, action):
        self.steps += 1
        state, reward, done = self.env.step(action)
        terminated = done
        truncated = self.steps >= self.max_steps
        self.context += f"> {action}\n{self.normalize_newlines(state.feedback)}"
        if terminated:      self.success += 1
        if truncated:       self.fail += 1

        if self.inference_type == 'ReAct':
            obs = self.context + "Think: "
        elif self.inference_type == 'ReAct-Im':
            obs = self.context + "Think: "
        elif self.inference_type == 'Act':
            obs = self.context + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "
        #print(obs)
        #sprint('-----------------------------------------------')
        return obs, reward, terminated, truncated, {}
        
    def add_think(self, think):
        if self.inference_type == 'ReAct':
            self.context += f"Think: {self.normalize_newlines(think)}"
            #print(self.context + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: ")
            #print('-----------------------------------------------')
            return self.context + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "
        elif self.inference_type == 'ReAct-Im':
            #print(self.context + f"Think: {self.normalize_newlines(think)}" + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: ")
            #print('-----------------------------------------------')
            return self.context + f"Think: {self.normalize_newlines(think)}" + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "

    def normalize_newlines(self, s):
        return s.replace("\n", " ") + '\n'
    
    def generate_new_game(self, options, seeds):
        options.seeds = seeds
        game_file, _ = textworld.make(options)
        self.env = textworld.start(game_file)
        self.env.reset()


class TextWorldEnvRL(gym.Env):
    def __init__(self, options, train_seed_num, max_steps=50):
        self.options = options # random.randint(1, 1000)
        self.max_steps = max_steps
        self.train_seed_num = train_seed_num

        self.observation_space = spaces.Text(
            max_length=4096,
            charset=TEXTWORLD_CHARSET
        )
        self.action_space = spaces.Text(
            max_length=256,
            charset=TEXTWORLD_CHARSET
        )

        self.steps = 0
        self.context = ""

        self.success = 0
        self.fail = 0
        self.eps = 1e-5

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.generate_new_game(self.options, random.randint(1, self.train_seed_num))
        state = self.env.state
        self.context = state.feedback[1210:] # remove TEXT WORLD
        obs = self.context + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "
        
        return obs, {}

    def step(self, action):
        self.steps += 1
        state, reward, done = self.env.step(action)
        terminated = done
        truncated = self.steps >= self.max_steps
        self.context += f"> {action}\n{self.normalize_newlines(state.feedback)}"
        if terminated:      self.success += 1
        if truncated:       self.fail += 1
        obs = self.context + f"Possible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "
        #print(obs)
        #sprint('-----------------------------------------------')
        return obs, reward, terminated, truncated, {}

    def normalize_newlines(self, s):
        return s.replace("\n", " ") + '\n'

    def generate_new_game(self, options, seeds):
        options.seeds = seeds
        game_file, _ = textworld.make(options)
        self.env = textworld.start(game_file)
        self.env.reset()
