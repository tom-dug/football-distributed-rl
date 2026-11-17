import model
import replay_buffer
import agent
import gfootball.env as football_env
from collections import deque
import torch

capacity = 80000
batch_size = 64
min_buffer_size = 2500

eps_start, eps_end, eps_decay_steps = 1.0, 0.05, 30000
max_episodes = 2500

env = football_env.create_environment(
    env_name="academy_empty_goal",
    representation="simple115v2",
    rewards="scoring,checkpoints",
    write_full_episode_dumps=True,
    dump_frequency=100,
    logdir='/app/dumps'
)

def get_epsilon(total_steps):
    epsilon = 1 + (eps_end - eps_start) / eps_decay_steps * total_steps
    return max(epsilon, eps_end)



total_steps = 0
DQNagent = agent.DQNAgent(state_dim=115, action_dim=19, lr=0.0003, gamma=0.99, target_update_freq=500)
buffer = replay_buffer.ReplayBuffer(capacity=capacity, state_dim=115)

past_scores = deque(maxlen=50)
past_ep_rewards = deque(maxlen=50)
past_ep_len = deque(maxlen=50)
past_losses = deque(maxlen=50)


for i in range(max_episodes):
    state = env.reset()
    
    done = False
    ep_reward = 0.0
    ep_len = 0
    ep_loss = 0
    while not done:
        eps = get_epsilon(total_steps)
        action = DQNagent.select_action(state, eps)
        next_state, reward, done, info = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        
        if len(buffer) >= min_buffer_size:
            sample = buffer.sample(batch_size)
            loss = DQNagent.update(sample)
            ep_loss += loss
            
        state = next_state
        ep_len += 1
        total_steps += 1
        ep_reward += reward
        
    past_scores.append(info["score_reward"])
    past_ep_rewards.append(ep_reward)
    past_ep_len.append(ep_len)
    past_losses.append(ep_loss/ep_len)
    
    if i % 20 == 0:
        print(f""" ===== Episode {i} =====
        - Total steps : {total_steps}
        - Epsilon : {eps}
        - Average score of last {len(past_scores)} episodes = {sum(past_scores) / len(past_scores):.4f}
        - Average reward of last {len(past_ep_rewards)} episodes = {sum(past_ep_rewards) / len(past_ep_rewards):.4f}
        - Average episode length of last {len(past_ep_len)} episodes = {sum(past_ep_len) / len(past_ep_len):.4f}
        - Average losses of last {len(past_losses)} episodes = {sum(past_losses) / len(past_losses):.4f}
        """)
        
        if sum(past_scores) / len(past_scores) >= 0.8 and len(past_scores) == 50:
            torch.save(DQNagent.q_net.state_dict(), "dqn_empty_goal.pt")
            break