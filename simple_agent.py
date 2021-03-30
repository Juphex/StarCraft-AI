from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path

from pysc2.lib import actions
from pysc2.agents import base_agent

import torch
import torch.nn as nn

from dqn import DQN
from replay_memory import ReplayMemory, Transition

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

'''Agent predicting actions for function 331 (Move_screen)'''

MOVE_FUNCTION_ID = 331


class Agent(base_agent.BaseAgent):
    def __init__(self, gamma=0.9, learning_rate=0.1, batch_size=64, batch_optim_steps=5, replay_memory_amount=10000,
                 random_action_threshold=0.9, exploration_steps=3000, target_update=100, policy_update=50,
                 path="model_weights.pt", train=True, model=DQN):
        super().__init__()
        self.GAMMA = gamma
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.BATCH_OPTIM_STEPS = batch_optim_steps
        # tuples of state, action, reward, next_state
        self.experience_replay_memory = ReplayMemory(replay_memory_amount)
        self.RANDOM_ACTION_THRESHOLD = random_action_threshold
        self.EXPLORATION_STEPS = exploration_steps
        self.TARGET_UPDATE = target_update
        self.POLICY_UPDATE = policy_update

        self.PATH = path

        self.TRAIN = train
        self.MODEL = model

        self.last_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        self.last_state = 0

        self.batch_loss = []

        self.possible_args = []
        for i in range(64):
            for j in range(64):
                self.possible_args.append([i, j])

        # Network
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

    def setup(self, obs_spec, action_spec):
        super(Agent, self).setup(obs_spec, action_spec)
        # function 331 2 outputs => multiply output[i] * 64 for real value
        self.policy_net = self.MODEL(38, 64 * 64).to(device)
        self.target_net = self.MODEL(38, 64 * 64).to(device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.LEARNING_RATE)

        if os.path.isfile(self.PATH):
            if not torch.cuda.is_available():
                self.policy_net.load_state_dict(torch.load(self.PATH, map_location=torch.device('cpu')))
                self.target_net.load_state_dict(torch.load(self.PATH, map_location=torch.device('cpu')))

            else:
                self.policy_net.load_state_dict(torch.load(self.PATH))
                self.target_net.load_state_dict(torch.load(self.PATH))

    # training loop, gathering data
    def step(self, obs):
        super(Agent, self).step(obs)
        # combine features into one tensor
        screens = self.get_screens(obs.observation.feature_screen,
                                   obs.observation.feature_minimap)
        reward = obs.reward
        if reward >= 1:
            reward = reward*5


        # push to replay memory when function 331 used in the last action
        if self.steps > 10 and self.last_action.function == MOVE_FUNCTION_ID and self.TRAIN is True:
            self.experience_replay_memory.push(self.last_state, self.last_action, screens,reward)
        selected_action = self.selectAction(obs)

        if self.steps % 100 == 0:
            print("steps: {}".format(self.steps))
            if len(self.batch_loss) > 0:
                print("loss: {}".format(sum(self.batch_loss) / len(self.batch_loss)))
            torch.save(self.policy_net.state_dict(), self.PATH)

        self.last_action = selected_action
        self.last_state = self.get_screens(obs.observation.feature_screen,
                                           obs.observation.feature_minimap)

        # optimization
        if self.steps > self.EXPLORATION_STEPS and self.steps % self.POLICY_UPDATE == 0 and self.TRAIN is True:
            self.batch_loss = []
            for i in range(self.BATCH_OPTIM_STEPS):
                self.optimize()

        # update target network
        if self.steps % self.TARGET_UPDATE == 0 and self.TRAIN is True:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return selected_action

    def optimize(self):
        # credits: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
        # not enough batches
        if len(self.experience_replay_memory) < self.BATCH_SIZE:
            return
        transitions = self.experience_replay_memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])

        # filter so that only actions using function 331 are used
        non_final_mask = list()
        non_final_next_states = list()
        batch_states = list()
        batch_actions = list()
        batch_rewards = list()
        for i in range(len(batch.action)):
            if (len(batch.action[i].arguments)) >= 2:
                # get index of possible_args
                batch_actions.append(torch.tensor(self.possible_args.index(batch.action[i].arguments[1])))
                batch_rewards.append(torch.tensor(batch.reward[i]))
                batch_states.append(batch.state[i].clone().detach())
                non_final_mask.append(tuple(map(lambda s: s is not None, batch.next_state[i])))
                non_final_next_states.append(batch.next_state[i])

        if len(non_final_next_states) < 1:
            return
        non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(non_final_next_states)

        state_batch = torch.cat(batch_states)
        batch_actions = torch.tensor(batch_actions)
        reward_batch = torch.tensor(batch_rewards)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch.to(device))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        valid_batch_size = len(reward_batch)
        next_state_values = self.target_net(non_final_next_states.to(device)).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.to(device) * self.GAMMA) + reward_batch.to(device).float()

        # add dimension for gather
        batch_actions = batch_actions.unsqueeze(0)
        state_action_values = torch.gather(state_action_values, 1, batch_actions.to(device))

        state_action_values = state_action_values.squeeze(0)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.batch_loss.append(loss)

    def selectAction(self, obs):
        selected_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        # exploration phase
        if self.steps < self.EXPLORATION_STEPS:
            return self.pickRandomAction(obs)

        # pick random action / noise
        if np.random.random() >= self.RANDOM_ACTION_THRESHOLD:
            selected_action = self.pickRandomAction(obs)
        else:
            with torch.no_grad():
                screens = self.get_screens(obs.observation.feature_screen,
                                           obs.observation.feature_minimap)

                outputs = self.policy_net.forward(screens.to(device))
                # check if function 331 is available
                highest_q_val_index = torch.argmax(outputs.detach())
                if MOVE_FUNCTION_ID in obs.observation.available_actions:
                    # multiply max size 64 with the outputvalue that ranges (0,1) inclusive on both side
                    args = [[0], self.map_out_to_action(highest_q_val_index)]
                    selected_action = actions.FunctionCall(MOVE_FUNCTION_ID, args)

        return selected_action

    def get_screens(self, feature_screen, feature_minimap):
        # shift dimensions => (channel, y, x)
        feature_screen = torch.tensor(feature_screen).permute(0, 2, 1)
        feature_minimap = torch.tensor(feature_minimap).permute(0, 2, 1)
        # concat screens to one tensor
        screens = torch.cat((feature_screen, feature_minimap), 0)
        screens = screens.unsqueeze(0)
        return screens

    '''out_value is equal to index  '''

    def map_out_to_action(self, out_value):
        return self.possible_args[out_value]

    def pickRandomAction(self, obs):
        if MOVE_FUNCTION_ID in obs.observation.available_actions:
            random_action_index = np.random.randint(0, 64 * 64)
            args = [[0], self.map_out_to_action(random_action_index)]
            return actions.FunctionCall(MOVE_FUNCTION_ID, args)
        # gathering information using current policy
        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
