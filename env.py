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
    def __init__(self, options, max_steps=50):
        options.seeds = random.randint(1, 1000)
        self.game_file = options
        self.max_steps = max_steps
        game_file, _  = textworld.make(options)
        self.env = textworld.start(game_file)

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

    def reset(self, seed=None, options=None):
        self.steps = 0
        state = self.env.reset()
        obs = self.normalize_newlines(state.feedback[1210:]) # remove TEXT WORLD
        self.context = obs
        return self.context + "Think:", {}

    def step(self, action):
        self.steps += 1
        state, reward, done = self.env.step(action)
        obs = state.feedback
        terminated = done
        truncated = self.steps >= self.max_steps
        self.context += f"> {action}\n{self.normalize_newlines(obs)}"
        return self.context + "Think:", reward, terminated, truncated, {}
        
    def add_think(self, think):
        self.context += f"Think: {self.normalize_newlines(think)}"
        return self.context + f"\n\nPossible Actions: {', '.join(self.env.state.possible_admissible_commands)}\nAction: "

    def normalize_newlines(self, s):
        return s.replace("\n", " ") + '\n'