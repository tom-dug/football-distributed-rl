import argparse
import grpc
import learner_pb2_grpc
import learner_pb2
from model import QNetwork
import gfootball.env as football_env
import numpy as np
import torch
import learner
import weights_utils


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
        

def parse_args():
    parser = argparse.ArgumentParser(description="Starts a new worker")
    parser.add_argument("--worker-id", type=str)
    parser.add_argument("--target", type=str, default="localhost:50051")
    parser.add_argument("--epsilon", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    worker = RLWorker(args.worker_id, args.target, args.epsilon)
    try:
        stub.


if __name__ == "__main__":
    main()