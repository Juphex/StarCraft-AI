from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.memory_with_rewards = []
        self.position_rewards = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        # results in iteration from 0 to len(capacity)
        self.position = (self.position + 1) % self.capacity

        # store rewards separatly
        if args[3] > 0:
            if len(self.memory_with_rewards) < self.capacity:
                self.memory_with_rewards.append(None)
            self.memory_with_rewards[self.position_rewards] = transition
            # results in iteration from 0 to len(capacity)
            self.position_rewards = (self.position_rewards + 1) % self.capacity

    # TODO REMOVE rewards batch ==
    def sample(self, batch_size):
        # rewards_amount = 0
        # rewards_batch_size = int(batch_size * 0.1)
        # random_rewards = []
        # if len(self.memory_with_rewards) > rewards_batch_size:
        #     rewards_amount = rewards_batch_size
        #     random_rewards = random.sample(self.memory_with_rewards, rewards_amount)
        # batch = random.sample(self.memory, batch_size - rewards_amount)
        # retVal = batch + random_rewards
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
