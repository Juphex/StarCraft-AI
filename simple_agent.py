from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pysc2.agents import base_agent
from pysc2.lib import actions

import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

torch.set_printoptions(profile="full")

from dqn import DQN
from replay_memory import ReplayMemory, Transition

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

'''Agent predicting actions for function 331 (Move_screen)'''

MOVE_FUNCTION_ID = 331


class Agent(base_agent.BaseAgent):
    def __init__(self, gamma=0.9, learning_rate=0.1, batch_size=64, batch_optim_steps=1, replay_memory_amount=10000,
                 epsilon=0.9, exploration_steps=3000, target_update=100, path="model_weights.pt",
                 train=True, stopping_condition=0.0001, score_multiplier=1):
        super().__init__()
        # Network
        self.PATH = path
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.TRAIN = train

        self.batch_loss = []

        self.GAMMA = gamma
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.BATCH_OPTIM_STEPS = batch_optim_steps
        self.EPSILON = epsilon
        self.EXPLORATION_STEPS = exploration_steps
        self.TARGET_UPDATE = target_update
        self.STOPPING_CONDITION = stopping_condition

        # tuples of state, action, reward, next_state
        self.experience_replay_memory = ReplayMemory(replay_memory_amount)

        self.last_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        self.last_state = 0

        self.possible_args = []

        self.RGB_SCREEN_SIZE = 64
        for i in range(0, self.RGB_SCREEN_SIZE, 1):
            for j in range(0, self.RGB_SCREEN_SIZE, 1):
                self.possible_args.append([i, j])

        self.last_episode = -1

        # grid search
        self.lastx = 0
        self.lasty = 0

        # store only a certain % of reward
        self.replay_buffer_switch_no_reward = False
        self.observed_rewards = 1
        self.observed_non_rewards = 1

        self.score_multiplier = score_multiplier
        self.rewards = 0
        self.rewards_per_episode = []

    def setup(self, obs_spec, action_spec):
        super(Agent, self).setup(obs_spec, action_spec)
        # function 331 2 outputs => multiply output[i] * 64 for real value
        self.policy_net = DQN(self.RGB_SCREEN_SIZE, self.RGB_SCREEN_SIZE, 6, 64 * 64).to(device)
        self.target_net = DQN(self.RGB_SCREEN_SIZE, self.RGB_SCREEN_SIZE, 6, 64 * 64).to(device)

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.LEARNING_RATE, eps=0.001,
                                             weight_decay=0.05)

        if os.path.isfile(self.PATH):
            if not torch.cuda.is_available():
                self.policy_net.load_state_dict(torch.load(self.PATH, map_location=torch.device('cpu')))

            else:
                self.policy_net.load_state_dict(torch.load(self.PATH))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # training loop, gathering data
    def step(self, obs):
        super(Agent, self).step(obs)
        # combine features into one tensor
        # screens = self.get_screens(obs.observation.feature_screen,
        #                            obs.observation.feature_minimap)
        screens = self.get_screens(obs.observation.rgb_screen,
                                   obs.observation.rgb_minimap)
        reward = obs.reward
        if self.steps > self.EXPLORATION_STEPS:
            self.rewards += reward
        if reward < 1:
            reward = -1

        # push to replay memory when function 331 used in the last action
        if self.steps > 10 and self.last_action.function == MOVE_FUNCTION_ID and self.TRAIN is True and \
                self.last_action.arguments[1] in self.possible_args:
            if reward > 0:
                self.experience_replay_memory.push(self.last_state, self.last_action.arguments[1], screens, reward)
                self.replay_buffer_switch_no_reward = True
                self.observed_rewards += 1
            elif self.replay_buffer_switch_no_reward:
                self.experience_replay_memory.push(self.last_state, self.last_action.arguments[1], screens, reward)
                self.replay_buffer_switch_no_reward = False
                self.observed_non_rewards += 1

            # store 1/3 reward samples
            if self.observed_rewards / (self.observed_rewards + self.observed_non_rewards) >= 1 / 3:
                self.replay_buffer_switch_no_reward = True

        if self.steps % 100 == 0:
            print("steps: {}".format(self.steps))
            torch.save(self.policy_net.state_dict(), self.PATH)
            if len(self.batch_loss) > 0:
                loss = sum(self.batch_loss) / len(self.batch_loss)
                print("loss: {}, learning rate: {}".format(loss,
                                                           self.LEARNING_RATE))
                if sum(self.batch_loss) / len(self.batch_loss) < self.STOPPING_CONDITION:
                    print("stop condition")
                    self.steps = 99999999999999999
                    return None

        if self.steps > self.EXPLORATION_STEPS:
            # optimization
            if self.TRAIN is True:
                self.batch_loss = []
                for i in range(self.BATCH_OPTIM_STEPS):
                    self.optimize()

            # update target network
            if self.steps % self.TARGET_UPDATE == 0 and self.TRAIN is True:
                print("copying weights to target net")
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # every 50k steps reduce random threshold to 10%
        if self.steps % 50000 and self.EPSILON < 0.9:
            self.EPSILON += 0.05

        selected_action = self.selectAction(obs, screens)

        if self.steps > self.EXPLORATION_STEPS:
            self.plotScore()

        self.last_action = selected_action
        self.last_state = screens

        return selected_action

    def selectAction(self, obs, screens):
        selected_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        # select the army unit at the start of each new episode:
        if self.episodes > self.last_episode:
            self.last_episode += 1
            self.rewards_per_episode.append(self.rewards / self.score_multiplier)
            self.rewards = 0
            return actions.FunctionCall(function=7, arguments=[[1]])

        # exploration phase
        if self.steps < self.EXPLORATION_STEPS and self.TRAIN is True:
            # return self.pickRandomAction(obs)
            return self.gridSearch(obs)

        # pick random action / epsilon greedy
        if np.random.random() >= self.EPSILON:
            selected_action = self.pickRandomAction(obs)
        else:
            with torch.no_grad():
                outputs = self.policy_net.forward(screens.to(device))
                # check if function 331 is available
                highest_q_val_index = torch.argmax(outputs)
                if MOVE_FUNCTION_ID in obs.observation.available_actions:
                    # multiply max size 64 with the outputvalue that ranges (0,1) inclusive on both side
                    args = [[0], self.map_index_to_action(highest_q_val_index)]
                    selected_action = actions.FunctionCall(MOVE_FUNCTION_ID, args)

        return selected_action

    def get_screens(self, feature_screen, feature_minimap):
        # shift dimensions => (channel, y, x)
        # IF FEATURE SCREEN
        # feature_screen = torch.tensor(feature_screen).permute(0, 2, 1)
        # feature_minimap = torch.tensor(feature_minimap).permute(0, 2, 1)

        feature_screen = torch.tensor(feature_screen).permute(2, 1, 0)
        feature_minimap = torch.tensor(feature_minimap).permute(2, 1, 0)

        # concat screens to one tensor
        screens = torch.cat((feature_screen, feature_minimap), 0)
        screens = screens.unsqueeze(0)
        return screens

    def map_index_to_action(self, out_value):
        return self.possible_args[out_value]

    def pickRandomAction(self, obs):
        if MOVE_FUNCTION_ID in obs.observation.available_actions:
            random_action_index = np.random.randint(0, 64 * 64)
            args = [[0], self.map_index_to_action(random_action_index)]
            return actions.FunctionCall(MOVE_FUNCTION_ID, args)

        # random action
        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)

    def gridSearch(self, obs):
        if MOVE_FUNCTION_ID in obs.observation.available_actions:
            # grid search
            if self.lasty + 3 < self.RGB_SCREEN_SIZE - 1:
                self.lasty += 3
                if self.lastx < self.RGB_SCREEN_SIZE - 1:
                    self.lastx += 1
                else:
                    self.lastx = 0
            # reset search
            else:
                self.lasty += 3
                self.lasty %= self.RGB_SCREEN_SIZE

            args = [[0], [self.lastx, self.lasty]]
            return actions.FunctionCall(MOVE_FUNCTION_ID, args)

        # random action
        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]

        return actions.FunctionCall(function_id, args)

    # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
    def plotScore(self):
        plt.figure(2)
        plt.clf()
        losses = torch.tensor(self.rewards_per_episode, dtype=torch.float)
        plt.title('Training...; lr={}'.format(self.LEARNING_RATE))
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.plot(losses.numpy())
        # Take 100 episode averages and plot them too
        if len(losses) >= 50:
            means = losses.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # uncomment to display plot
        # if is_ipython:
        #     display.clear_output(wait=True)
        #     display.display(plt.gcf())

        # store plot
        plt.savefig("loss_plot_{}.png".format(self.LEARNING_RATE))

    def optimize(self):
        ''' # zero gradients
         self.optimizer.zero_grad()

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
                 # get index of action/possible_args
                 batch_actions.append(torch.tensor(self.possible_args.index(batch.action[i].arguments[1])))
                 batch_rewards.append(torch.tensor(batch.reward[i]))
                 batch_states.append(batch.state[i].clone().detach())
                 non_final_mask.append(tuple(map(lambda s: s is not None, batch.next_state[i])))
                 non_final_next_states.append(batch.next_state[i])

         if len(non_final_next_states) < 1:
             return
         # non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)
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
         # MAX Q VALUE FOR NEXT STATE
         next_state_values = self.target_net(non_final_next_states.to(device)).detach().max(1)[0]
         # Compute the expected Q values
         # expected_state_action_values = reward_batch.to(device).float() + (next_state_values.to(device) * self.GAMMA)
         expected_state_action_values = reward_batch.to(device) + (next_state_values.to(device) * self.GAMMA)

         # add dimension for gather
         batch_actions = batch_actions.unsqueeze(0)
         # gather actions values at the indices from batch_actions
         state_action_values = torch.gather(state_action_values, 1, batch_actions.to(device))

         expected_state_action_values = expected_state_action_values.unsqueeze(0)

         # Compute Huber loss
         criterion = nn.SmoothL1Loss()
         loss = criterion(state_action_values, expected_state_action_values)

         # Optimize the model
         loss.backward()
         # gradient clipping
         for param in self.policy_net.parameters():
             param.grad.data.clamp_(-1, 1)
         self.optimizer.step()
         self.batch_loss.append(loss)
 '''

        # TODO aus tutorial
        if len(self.experience_replay_memory) < self.BATCH_SIZE:
            return
        transitions = self.experience_replay_memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(
            tuple([torch.tensor(self.possible_args.index(action)).unsqueeze(0) for action in batch.action])).unsqueeze(
            1)
        reward_batch = torch.tensor(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch.to(device)).gather(1, action_batch.to(device))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.to(device)).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.to(device)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.batch_loss.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
