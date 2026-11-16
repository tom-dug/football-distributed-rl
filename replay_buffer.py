import numpy as np
from numpy import float32


class ReplayBuffer():
    def __init__(self, capacity: int, state_dim: int = 115):
        self.states = np.zeros((capacity, state_dim), float32)
        self.actions = np.zeros((capacity,), np.int64)
        self.rewards = np.zeros((capacity,), float32)
        self.next_states = np.zeros((capacity, state_dim), float32)
        self.dones = np.zeros((capacity,), float32)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity
        
    def __len__(self):
        return self.size
    
    def push(self, state, action, reward, next_state, done) -> None:
        """
        Pushes a set of transitions to the replay buffer
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a random sample of the replay buffer
        """
        indexes = np.random.randint(0, self.size, batch_size)
            
        return (self.states[indexes],
                self.actions[indexes],
                self.rewards[indexes],
                self.next_states[indexes],
                self.dones[indexes])


if __name__ == "__main__":
    buffer = ReplayBuffer(5)
    for i in range(7):
        buffer.push(np.zeros(115), 0, 0.1*i, np.zeros(115), 0)
    print(len(buffer))
    print(buffer.rewards)
    
    samp = buffer.sample(3)
    print(samp[0].shape, samp[1].shape, samp[2].shape, samp[3].shape, samp[4].shape)