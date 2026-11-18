import numpy as np
from numpy import float32
from threading import Lock, Thread


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


class ThreadSafeReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_dim = 115):
        super().__init__(capacity, state_dim)
        self.lock = Lock()
    
    def __len__(self):
        with self.lock:
            return super().__len__()
    
    def push(self, state, action, reward, next_state, done):
        with self.lock:
            super().push(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int):
        with self.lock:
            batch = super().sample(batch_size)
            return batch
    
    def push_batch(self,
                   states: np.ndarray,
                   actions: np.ndarray,
                   rewards: np.ndarray,
                   next_states: np.ndarray,
                   dones: np.ndarray):
        n = states.shape[0]
        
        if n == 0:
            return
        
        with self.lock:
            if n >= self.capacity:
                self.states[:] = states[n - self.capacity:]
                self.actions[:] = actions[n - self.capacity:]
                self.rewards[:] = rewards[n - self.capacity:]
                self.next_states[:] = next_states[n - self.capacity:]
                self.dones[:] = dones[n - self.capacity:]
                self.ptr = 0
                self.size = self.capacity
            
            elif n + self.ptr <= self.capacity:
                end = self.ptr + n
                self.states[self.ptr:end] = states
                self.actions[self.ptr:end] = actions
                self.rewards[self.ptr:end] = rewards
                self.next_states[self.ptr:end] = next_states
                self.dones[self.ptr:end] = dones
                self.ptr = (self.ptr + n) % self.capacity
                self.size = min(self.size + n, self.capacity)
                
            else:
                first = self.capacity - self.ptr
                rest = n - first
                self.states[self.ptr:] = states[:first]
                self.actions[self.ptr:] = actions[:first]
                self.rewards[self.ptr:] = rewards[:first]
                self.next_states[self.ptr:] = next_states[:first]
                self.dones[self.ptr:] = dones[:first]
                
                self.states[:rest] = states[first:]
                self.actions[:rest] = actions[first:]
                self.rewards[:rest] = rewards[first:]
                self.next_states[:rest] = next_states[first:]
                self.dones[:rest] = dones[first:]
                
                self.ptr = rest
                self.size = self.capacity



if __name__ == "__main__":
    """buffer = ReplayBuffer(5)
    for i in range(7):
        buffer.push(np.zeros(115), 0, 0.1*i, np.zeros(115), 0)
    print(len(buffer))
    print(buffer.rewards)
    
    samp = buffer.sample(3)
    print(samp[0].shape, samp[1].shape, samp[2].shape, samp[3].shape, samp[4].shape)"""
    t1 = Thread()
    t2 = Thread()
    