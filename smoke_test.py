import gfootball.env as football_env

env = football_env.create_environment(
    env_name="academy_empty_goal",
    representation="simple115v2",
    rewards="scoring,checkpoints",
    render=False
)

obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
print("Episode finished. Score:", info["score_reward"], "total reward:", total_reward)
env.close()