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
                 epsilon=0.1, exploration_steps=3000, target_update=100, path="model_weights.pt",
                 train=True, stopping_condition=0.0001, score_multiplier=1, rgb_screen_size=64):
        super().__init__()
        ''' network '''
        # path to save
        self.PATH = path
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        # bool variable, when false network does not train
        self.TRAIN = train

        self.batch_loss = []

        # gamme of update function for Q-values
        self.GAMMA = gamma
        # learning rate of optimizer
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        # how many times the model is optimized using one batch
        self.BATCH_OPTIM_STEPS = batch_optim_steps

        # when to update the target, i.e. every 10 steps
        self.TARGET_UPDATE = target_update
        self.STOPPING_CONDITION = stopping_condition

        ''' e-greedy '''
        # epsilon is the inverted probability. Example: 0.9 => 1-0.9 = 0.1 => 10 % probability for random picking
        self.EPSILON = 1 - epsilon
        self.LIVE_EPSILON = 1 - 0.9

        ''' agent '''
        self.EXPLORATION_STEPS = exploration_steps

        # setup replay memory; tuples of state, action, reward, next_state
        self.experience_replay_memory = ReplayMemory(replay_memory_amount)

        # store only a certain % of reward
        self.replay_buffer_switch_no_reward = False
        self.observed_rewards = 1
        self.observed_non_rewards = 1

        self.last_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        self.last_state = 0

        # score multiplier
        self.score_multiplier = score_multiplier
        # init rewards
        self.rewards = 0
        self.rewards_per_episode = []

        '''action vector for the function'''
        self.possible_args = []

        self.RGB_SCREEN_SIZE = rgb_screen_size
        # create entry for every possible action
        for i in range(0, self.RGB_SCREEN_SIZE, 1):
            for j in range(0, self.RGB_SCREEN_SIZE, 1):
                self.possible_args.append([i, j])

        self.last_episode = -1

        # variables for grid search
        self.lastx = 0
        self.lasty = 0

    # sets up network and optimizer
    def setup(self, obs_spec, action_spec):
        super(Agent, self).setup(obs_spec, action_spec)
        self.policy_net = DQN(self.RGB_SCREEN_SIZE, self.RGB_SCREEN_SIZE, 6, 64 * 64).to(device)
        self.target_net = DQN(self.RGB_SCREEN_SIZE, self.RGB_SCREEN_SIZE, 6, 64 * 64).to(device)

        # use adam beta1 0.9 beta2 0.999 eppsilon 1e-8
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.LEARNING_RATE, eps=0.001,
                                             weight_decay=0.05)

        # read model when there is one in the given path
        if os.path.isfile(self.PATH):
            if not torch.cuda.is_available():
                self.policy_net.load_state_dict(torch.load(self.PATH, map_location=torch.device('cpu')))

            else:
                self.policy_net.load_state_dict(torch.load(self.PATH))
        # initalize target network with the same network weights as the policy network
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

        # add reward to list when training
        if self.steps > self.EXPLORATION_STEPS:
            self.rewards += reward
        # give penalty for no rewards
        if reward < 1:
            reward = -1

        # push to replay memory when function 331 used in the last action
        if self.steps > 10 and self.last_action.function == MOVE_FUNCTION_ID and self.TRAIN is True and \
                self.last_action.arguments[1] in self.possible_args:
            # when ever there is a reward > 0 add it to the memory
            if reward > 0:
                self.experience_replay_memory.push(self.last_state, self.last_action.arguments[1], screens, reward)
                self.replay_buffer_switch_no_reward = True
                self.observed_rewards += 1
            # when the reward is <= 0 check if the "gate" is open
            elif self.replay_buffer_switch_no_reward:
                self.experience_replay_memory.push(self.last_state, self.last_action.arguments[1], screens, reward)
                self.replay_buffer_switch_no_reward = False
                self.observed_non_rewards += 1

            # enable the "gate" when the ratio of 1/3 reward samples is true
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
        if self.steps % 50000 and self.LIVE_EPSILON < self.EPSILON:
            self.LIVE_EPSILON += 0.05

        selected_action = self.selectAction(obs, screens)

        if self.steps > self.EXPLORATION_STEPS:
            self.plotScore()

        self.last_action = selected_action
        self.last_state = screens

        return selected_action

    # selects action
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
        # when LIVE_EPSILON = 0.8 => 20% chance of taking random action
        if np.random.random() >= self.LIVE_EPSILON:
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

    # concats two screens into one tensor
    def get_screens(self, feature_screen, feature_minimap):
        # shift dimensions => (channel, y, x)
        # IF FEATURE SCREEN
        # feature_screen = torch.tensor(feature_screen).permute(0, 2, 1)
        # feature_minimap = torch.tensor(feature_minimap).permute(0, 2, 1)

        feature_screen = torch.tensor(feature_screen).permute(2, 1, 0)
        feature_minimap = torch.tensor(feature_minimap).permute(2, 1, 0)

        # concat screens to one tensor
        screens = torch.cat((feature_screen, feature_minimap), 0)

        # screens = feature_minimap
        screens = screens.unsqueeze(0)
        return screens

    # get coordinates from an index
    def map_index_to_action(self, out_value):
        return self.possible_args[out_value]

    # picks random action
    # pick action from MOVE_FUNCTION_ID or fully random when not available
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

    # performs one step of grid search
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
        plt.ylabel('Score')
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
        # from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
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
        # Compute the expected (target) Q values
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
