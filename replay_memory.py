from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        # results in iteration from 0 to len(capacity)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        reward_size = 0
        reward_batch = []

        return random.sample(self.memory, batch_size - reward_size) + reward_batch

    def __len__(self):
        return len(self.memory)
