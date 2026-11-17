import gfootball.env as football_env
from agent import DQNAgent
import torch
from evaluate import evaluate

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
agent.q_net.load_state_dict(torch.load("dqn_empty_goal.pt", map_location="cpu"))

n_episodes = 5

state = env.reset()
done = False
total_reward = 0.0

results = evaluate(agent=agent, env=env, n_episodes=n_episodes)
    
print(f"""{n_episodes} episodes finished:
      - Goals scored: {results["goals_scored"]}
      - Goal rate: {results["goal_rate"]}
      - Own goals scored: {results["own_goals_scored"]}
      """)
env.close()