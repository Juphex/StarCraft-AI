from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as numpy
import torch

from pysc2.lib import actions
from pysc2.agents import base_agent
from pysc2.lib import features

from networks import Model

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

# pytorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class Agent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.alpha = 0.4
        self.decay = 1
        self.BatchesPerStep = 20
        self.UpdatesPerBatch = 5
        self.BatchSize = 20
        # tuples of state, action, reward, next_state
        self.experience_replay_memory = []
        self.StepsBeforeBatching = 100
        # TODO initialize network parameters

        self.last_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        self.last_state = 0
        self.last_reward = 0

        # Network
        self.model = None

    def setup(self, obs_spec, action_spec):
        super(Agent, self).setup(obs_spec, action_spec)
        print(len(self.action_spec.functions))
        print(self.obs_spec)

        self.model = Model((64, 64, 38), 1).to(device)

    def step(self, obs):
        super(Agent, self).step(obs)
        # save reward (for this state)
        # check state
        # predict action
        selected_action = None
        # features.MINIMAP_FEATURES
        # features.SCREEN_FEATURES
        if self.steps == 2:
            print(features.SCREEN_FEATURES.selected)

        # actions.ValidActions
        self.model.forward(obs.observation.feature_screen, obs.observation.feature_minimap)

        if self.steps < self.StepsBeforeBatching:
            # gathering information using current policy
            function_id = numpy.random.choice(obs.observation.available_actions)
            args = [[numpy.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
            selected_action = actions.FunctionCall(function_id, args)
        else:
            function_id = numpy.random.choice(obs.observation.available_actions)
            args = [[numpy.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
            selected_action = actions.FunctionCall(function_id, args)
        # save trajectory if step >0
        self.experience_replay_memory.append(())
        return selected_action
