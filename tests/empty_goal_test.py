import gfootball.env as football_env
from agent import DQNAgent
import torch

env = football_env.create_environment(
    env_name="academy_empty_goal",
    representation="simple115v2",
    rewards="scoring,checkpoints",
    render=True,
    write_full_episode_dumps=True,
    write_video=True,
    dump_frequency=1,
    logdir='/app/videos'
)

agent = DQNAgent()
agent.q_net.load_state_dict(torch.load("dqn_empty_goal.pt"))

state = env.reset()
done = False
total_reward = 0.0

while not done:
    action = agent.select_action(state, 0)
    next_state, reward, done, info = env.step(action)
    state = next_state
    total_reward += reward
    
print("Episode finished. Score:", info["score_reward"], "total reward:", total_reward)
env.close()