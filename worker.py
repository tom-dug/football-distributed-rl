import argparse
import grpc
import learner_pb2_grpc
import learner_pb2
from model import QNetwork
import gfootball.env as football_env
import numpy as np
import torch
import weights_utils
from collections import deque
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 

torch.set_num_threads(1)

class RLWorker:
    def __init__(self,
                 worker_id:str,
                 target:str = "localhost:50051",
                 epsilon:float = 0.2,
                 env_name: str = "academy_empty_goal",
                 batch_push_size: int = 64,
                 sync_freq: int = 100):
        self.worker_id = worker_id
        self.target = target
        self.epsilon = epsilon
        self.env_name = env_name
        self.batch_push_size = batch_push_size
        self.sync_freq = sync_freq
        self.local_version = -1
        self.q_net = QNetwork()
        self.q_net.eval()
        self.env = football_env.create_environment(
            env_name=self.env_name,
            representation="simple115v2",
            rewards="scoring,checkpoints",
            write_full_episode_dumps=False
        )
        self.action_dim = self.env.action_space.n
        
        self.channel = grpc.insecure_channel(target)
        self.stub = learner_pb2_grpc.LearnerServiceStub(self.channel)
        self.local_batch: list[learner_pb2.Transition] = []
        
    def select_action(self, state: np.ndarray) -> int:
        if self.epsilon > np.random.uniform(0, 1):
            return np.random.randint(0, self.action_dim)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_output = self.q_net(state_tensor)
            return q_output.argmax(dim=1).item()
    
    def pull_weights(self) -> bool:
        request = learner_pb2.WeightsRequest(worker_id=self.worker_id, current_version=self.local_version)
        response = self.stub.GetWeights(request)
        if response.updated:
            self.local_version = response.model_version
            new_state_dict = weights_utils.proto_to_state_dict(response.tensors)
            self.q_net.load_state_dict(new_state_dict)
            return True
        else:
            return False
        
    def push_transitions(self):
        if len(self.local_batch) == 0:
            return None
        else:
            batch = learner_pb2.TransitionBatch(workerid = self.worker_id,
                                        model_version = self.local_version,
                                        transition=self.local_batch)
            ack = self.stub.PushTransitions(batch)
            self.local_batch.clear()
            return ack
        
    def run(self, max_episodes:int | None = None):
        self.pull_weights()
        total_steps = 0
        episode = 0
        last_buffer_size = 0
        past_scores = deque(maxlen=50)
        past_rewards = deque(maxlen=50)
        while max_episodes is None or episode < max_episodes:
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                transition = learner_pb2.Transition(state=state.tolist(),
                                                    action=int(action),
                                                    reward=float(reward),
                                                    next_state=next_state.tolist(),
                                                    done=bool(done))
                self.local_batch.append(transition)
                
                if len(self.local_batch) >= self.batch_push_size:
                    ack = self.push_transitions()
                    if ack is not None:
                        last_buffer_size = ack.buffer_size
                
                if total_steps % self.sync_freq == 0 and total_steps > 0:
                    self.pull_weights()
                
                state = next_state
                ep_reward += reward
                total_steps += 1
            
            episode += 1
            past_scores.append(info["score_reward"])
            past_rewards.append(ep_reward)
            
            if episode % 50 == 0:
                print(f"""
        ***** Worker {self.worker_id} *****
        - Episode: {episode}
        - Total steps: {total_steps}
        - Local version: {self.local_version}
        - Last buffer size: {last_buffer_size}
        - Average score in past 50 games: {sum(1 for s in past_scores if s > 0) / len(past_scores):.4f}
        - Average reward in past 50 games: {sum(past_rewards) / len(past_rewards):.4f}
        """)
        self.push_transitions()
    
    def close(self):
        self.env.close()
        self.channel.close()

    
        

def parse_args():
    parser = argparse.ArgumentParser(description="Starts a new worker")
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--target", type=str, default="localhost:50051")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--env", type=str, default="academy_empty_goal")
    parser.add_argument("--batch-push-size", type=int, default=64)
    parser.add_argument("-sync-freq", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    worker = RLWorker(args.worker_id, args.target, args.epsilon)
    try:
        print(torch.get_num_threads())
        worker.run()
    except KeyboardInterrupt:
        print(f"Arrêt du worker {worker.worker_id} en cours...")
    finally:
        worker.close()


if __name__ == "__main__":
    main()