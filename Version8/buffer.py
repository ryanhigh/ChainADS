import random

class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.nextstates = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.nextstates[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def sample_batch(self, batch_size=10):
        # Sample a random batch from the buffer
        batch_indices = random.sample(range(len(self.states)), batch_size)
        
        # Gather the samples using the indices
        states = [self.states[i] for i in batch_indices]
        actions = [self.actions[i] for i in batch_indices]
        nextstates = [self.nextstates[i] for i in batch_indices]
        rewards = [self.rewards[i] for i in batch_indices]
        is_terminals = [self.is_terminals[i] for i in batch_indices]
        
        return states, actions, nextstates, rewards, is_terminals

