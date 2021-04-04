from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.env import run_loop
from pysc2.env import available_actions_printer
from absl import flags
from absl import app

# from agent import Agent
from simple_agent import Agent

# notebook error
# import sys
#
# sys.argv = sys.argv[:1]
FLAGS = flags.FLAGS

# dunno, if flags already exist error: restart kernel
# flags info: https://github.com/deepmind/pysc2/blob/master/pysc2/bin/agent.py
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_float("learning_rate", 1e-10, "Learning rate.")
flags.DEFINE_string("path", "model_weights.pt", "Path and filename of the network weights to be loaded and saved.")


# notebook error
# FLAGS(sys.argv)


def main(argv):
    print('non-flag arguments:', argv)

    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            players=[sc2_env.Agent(race=sc2_env.Race.terran)],
            agent_interface_format=[features.parse_agent_interface_format(
                # feature_screen=(64, 64),
                # feature_minimap=(64, 64),
                rgb_screen=(64, 64),
                rgb_minimap=(64, 64))],
            step_mul=4,
            disable_fog=True,
            visualize=True) as env:
        # list of players.
        agents = [
            Agent(batch_size=16, exploration_steps=10000,
                  replay_memory_amount=20000, gamma=0.8, learning_rate=FLAGS.learning_rate,
                  random_action_threshold=0.9, path=FLAGS.path)]
        env = available_actions_printer.AvailableActionsPrinter(env)
        run_loop.run_loop(agents, env, max_episodes=200)


if __name__ == "__main__":
    app.run(main)
