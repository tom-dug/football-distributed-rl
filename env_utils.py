import gfootball.env as football_env


env = football_env.create_environment(
    env_name="academy_empty_goal",
    representation="simple115v2",
    rewards="scoring,checkpoints",
    write_full_episode_dumps=True,
    logdir="/app/videos"
)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    print(f"Observation space : {env.observation_space.shape}")
    print(f"Action space : {env.action_space.n}")
    obs = env.reset()
    print(f"Observation resset : {obs.dtype}, {obs.shape}")