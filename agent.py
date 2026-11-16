from model import QNetwork
from torch import optim, nn, as_tensor, no_grad, gather, float32
import numpy as np
import replay_buffer

class DQNAgent():
    def __init__(self, state_dim=115, action_dim=19, lr=3e-4, gamma=0.99, target_update_freq=500):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.learn_steps = 0
        self.action_dim = action_dim
        self.target_update_freq = target_update_freq
        self.gamma = gamma
    
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if epsilon > np.random.uniform(0, 1):
            return np.random.randint(0, self.action_dim)
        else:
            state_tensor = as_tensor(state, dtype=float32).unsqueeze(0)
            with no_grad():
                q_output = self.q_net(state_tensor)
            return q_output.argmax(dim=1).item()
    
    def update(self, batch: tuple) -> float:
        states_, actions_, rewards_, next_states_, dones_ = batch
        states = as_tensor(states_)
        next_states = as_tensor(next_states_)
        actions = as_tensor(actions_).unsqueeze(1)
        rewards = as_tensor(rewards_).unsqueeze(1)
        dones = as_tensor(dones_).unsqueeze(1)
        q_values = self.q_net(states).gather(1, actions)
        
        with no_grad():
            next_q = self.q_net(next_states).max(dim=1, keepdim=True).values
        targets = rewards + self.gamma * (1.0 - dones) * next_q     # Bellman's equation
        loss = self.loss_fn(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()
        

if __name__ == "__main__":
    dqn = DQNAgent()
    print(dqn.select_action(np.zeros(115), 0.9))
    buffer = replay_buffer.ReplayBuffer(100)
    for i in range(7):
        buffer.push(np.zeros(115), 0, 0.1*i, np.zeros(115), 0)
    batch = buffer.sample(64)
    print(dqn.update(batch))