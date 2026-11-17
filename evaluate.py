from agent import DQNAgent


def evaluate(agent: DQNAgent, env, n_episodes: int):
    """
    Plays n episodes with epsilon null and without learning.
    """
    agent.q_net.eval()
    past_eps_rewards = []
    past_eps_lens = []
    goals_scored = 0
    own_goals_scored = 0
    for i in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        
        while not done:
            action = agent.select_action(state, 0)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            ep_reward += reward
            ep_len += 1
        
        past_eps_rewards.append(ep_reward)
        past_eps_lens.append(ep_len)
        goal = info["score_reward"]
        if goal == 1:
            goals_scored += 1
        if goal == -1:
            own_goals_scored += 1
        
    agent.q_net.train()
    avg_rewards = sum(past_eps_rewards) / len(past_eps_rewards)
    avg_ep_len = sum(past_eps_lens) / len(past_eps_lens)
    return {"avg_rewards": avg_rewards,
            "avg_episode_len": avg_ep_len,
            "goals_scored": goals_scored,
            "own_goals_scored": own_goals_scored,
            "goal_rate": goals_scored/n_episodes
            }