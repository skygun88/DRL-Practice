from collections import deque
import torch.optim as optim

OPTIMIZER = {'rmsprop': optim.RMSprop}

class Trainer:
    def __init__(self, memory_capacity, optimizer, minibatch_size) -> None:
        self.replay_memory = deque([], maxlen=memory_capacity)
        self.optimizer_name = optimizer
        self.minibatch_size = minibatch_size

    def train(self, model):