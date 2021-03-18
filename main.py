from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.env import run_loop
from pysc2.env import available_actions_printer
from absl import flags

# from agent import Agent
from simple_agent import Agent

import sys
sys.argv = sys.argv[:1]
FLAGS = flags.FLAGS
# dunno, if flags already exist error: restart kernel
# flags info: https://github.com/deepmind/pysc2/blob/master/pysc2/bin/agent.py
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
FLAGS(sys.argv)


if __name__ == "__main__":
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            players=[sc2_env.Agent(race=sc2_env.Race.protoss)],
            agent_interface_format=[features.parse_agent_interface_format(
                feature_screen=(64, 64),
                feature_minimap=(64, 64),
                rgb_screen=None,
                rgb_minimap=None)],
            step_mul=4,
            disable_fog=True) as env:
        # list of players.
        agents = [Agent()]
        env = available_actions_printer.AvailableActionsPrinter(env)
        run_loop.run_loop(agents, env, max_episodes=100)
