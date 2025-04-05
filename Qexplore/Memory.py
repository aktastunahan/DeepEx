import random
from collections import deque

class Memory:
    memory = deque()
    def __init__(self, maxlen=2000):
        self.memory = deque(maxlen=maxlen)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        if self.can_sample(batch_size):
            return random.sample(self.memory, min(batch_size, len(self.memory)))
        else:
            return None

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)