from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.env import run_loop
from pysc2.env import available_actions_printer
from absl import flags
from absl import app

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# from agent import Agent
from simple_agent import Agent

FLAGS = flags.FLAGS

# flags info: https://github.com/deepmind/pysc2/blob/master/pysc2/bin/agent.py
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
#flags.DEFINE_string("map", "CollectMineralShards", "Name of a map to use.")
flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
# 2,5e-11, 2,5e-10, 2,5e-9, 2,5e-8, 2,5e-7 2,5e-6 2,5e-5, 2,5e-4
flags.DEFINE_float("learning_rate", 2e-05, "Learning rate.")
flags.DEFINE_string("path", "model_weights.pt", "Path and filename of the network weights to be loaded and saved.")


def main(argv):
    print('non-flag arguments:', argv)

    lr = FLAGS.learning_rate
    path = FLAGS.path
    score_multiplier = 5

    # for lr in [1e-1, 1e-2, 1e-3,  1e-4, 1e-9, 1e-10, 1e-11, 1e-12, 1e-14]:
    #     path = "{}.pt".format(lr)
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            players=[sc2_env.Agent(race=sc2_env.Race.terran)],
            agent_interface_format=[features.parse_agent_interface_format(
                # feature_screen=(64, 64),
                # feature_minimap=(64, 64),
                rgb_screen=(64, 64),
                rgb_minimap=(64, 64))],
            step_mul=20,
            disable_fog=True,
            visualize=True,
            score_multiplier=score_multiplier
    ) as env:
        # list of players.
        agents = [
            Agent(batch_size=16, exploration_steps=10000,
                  replay_memory_amount=10000, gamma=0.99, learning_rate=lr,
                  epsilon=0.6, path=path, score_multiplier=score_multiplier)]
        env = available_actions_printer.AvailableActionsPrinter(env)
        run_loop.run_loop(agents, env, max_episodes=1000)

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    app.run(main)
