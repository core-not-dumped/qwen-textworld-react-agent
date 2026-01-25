import textworld
from textworld import EnvInfos

options = textworld.GameOptions()
options.seeds = 1234
options.nb_objects = 5
options.quest_length = 2
game_file, _ = textworld.make(options)  # Generate a random game.
env = textworld.start(game_file)  # Load the game.
game_state = env.reset()  # Start a new game.
a = env.step("look")
print(a[0])

