from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch

from pysc2.lib import actions
from pysc2.agents import base_agent
from pysc2.lib import features
from torch import nn

from dqn import Model

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
        self.StepsBeforeBatching = 10

        self.last_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        self.last_state = 0
        self.last_reward = 0

        # Network
        self.model = None

    def setup(self, obs_spec, action_spec):
        super(Agent, self).setup(obs_spec, action_spec)
        self.model = Model(38, len(self.action_spec.functions)).to(device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.2)

    def step(self, obs):
        super(Agent, self).step(obs)
        selected_action = None
        function_id = None

        self.model.zero_grad()
        self.optimizer.zero_grad()

        if self.steps % 100 == 0:
            print(self.steps)

        # save trajectory if step >0
        if self.steps >= 10:
            self.experience_replay_memory.append(
                (self.last_state, self.last_action, self.last_reward, [obs.observation.feature_screen,
                                                                       obs.observation.feature_minimap]))

            pred_q_value, args1, args2, args3, args4, args5 = self.model.forward(self.last_state[0],
                                                                                 self.last_state[1])
            pred_action, args1, args2, args3, args4, args5 = self.model.forward(obs.observation.feature_screen,
                                                                                obs.observation.feature_minimap)
            # q values for possible actions
            q_values = pred_action.detach().numpy()[:, obs.observation.available_actions]
            # re-map output to available_actions (indices lost) and get function_id with max q value
            function_id = obs.observation.available_actions[np.argmax(q_values)]
            target = self.last_reward * 1000 + 0.9 * pred_action.detach().numpy()[:, function_id]

            arg_values = [args1.detach().numpy()[0], args2.detach().numpy()[0],
                          args3.detach().numpy()[0], args4.detach().numpy()[0],
                          args5.detach().numpy()[0]]

            args = [0, 0, 0, 0, 0]
            # print(self.last_action[0])
            # print(self.last_action[1])
            if self.last_action[1] is not None and self.last_action[0] is not None:
                iterator = 0
                for arg in self.last_action[1]:
                    for sub_arg in arg:
                        args[iterator] = sub_arg
                        iterator += 1

                # divide by size => range(0,1) which is the output of the neural network
                iterator = 0
                for arg in self.action_spec.functions[self.last_action[0]].args:
                    for size in arg.sizes:
                        if args[iterator] != 0:
                            args[iterator] /= size

            predicted_args = torch.cat(
                (args1, args2, args3, args4,
                 args5,
                 ))
            criterion = nn.MSELoss()
            criterion2 = nn.MSELoss()

            target_tensor = torch.ones((1,pred_action.size()[1]))
            print(target_tensor)
            print(function_id)
            target_tensor[function_id] = pred_action[np.argmax(q_values)]
            loss = criterion(pred_q_value, target_tensor) + criterion2(predicted_args,
                                                                              torch.tensor(args).float())
            loss.backward()
            self.optimizer.step()


        if self.steps < self.StepsBeforeBatching:
            selected_action = self.pickRandomAction(obs)
        else:
            # # set args
            args = []
            for arg in self.action_spec.functions[function_id].args:
                temp_args = []
                for size in arg.sizes:
                    # round everything to int
                    #subtract because size is exclusive and should not be reached in some cases
                    temp_args.append(int(np.around((arg_values.pop(0)) * size) - 1))
                args.append(temp_args)
            selected_action = actions.FunctionCall(function_id, args)

            if np.random.random() >= 0.9:
                print("random action")
                selected_action = self.pickRandomAction(obs)

        self.last_action = [function_id, selected_action.arguments]
        self.last_reward = obs.reward
        self.last_state = [obs.observation.feature_screen, obs.observation.feature_minimap]
        return selected_action

    def pickRandomAction(self, obs):
        # gathering information using current policy
        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
