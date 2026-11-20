import learner_pb2
import learner_pb2_grpc
from replay_buffer import ThreadSafeReplayBuffer
from threading import Lock
import numpy as np
import threading
from agent import DQNAgent
import time
import grpc
from concurrent import futures
from collections import deque
from weights_utils import state_dict_to_proto, proto_to_state_dict


class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    def __init__(self, buffer: ThreadSafeReplayBuffer, initial_weights: list[learner_pb2.TensorData]):
        super().__init__()
        self.buffer = buffer
        self.version = 0
        self.cached_weights = initial_weights
        self.weights_lock = Lock()
    
    def update_weights(self, new_weights: list[learner_pb2.TensorData]) -> None:
        with self.weights_lock:
            self.cached_weights = new_weights
            self.version += 1
    
    def PushTransition(self, request, context):
        if len(request.transition) > 0:
            transition = request.transition
            states_, actions_, rewards_, next_states_, dones_ = [], [], [], [], []
            for t in transition:
                states_.append(t.state)
                actions_.append(t.action)
                rewards_.append(t.reward)
                next_states_.append(t.next_state)
                dones_.append(int(t.done))
            states = np.array(states_, np.float32)
            actions = np.array(actions_, np.int64)
            rewards = np.array(rewards_, np.float32)
            next_states = np.array(next_states_, np.float32)
            dones = np.array(dones_, np.float32)
            self.buffer.push_batch(states, actions, rewards, next_states, dones)
            with self.weights_lock:
                version = self.version
            size = len(self.buffer)
            return learner_pb2.PushAck(accepted=True, current_model_version=version, buffer_size=size)
        else:
            return learner_pb2.PushAck(accepted=False)
            
    def GetWeights(self, request, context):
        with self.weights_lock:
            version = self.version
            weights = self.cached_weights
        if request.current_version >= version:
            return learner_pb2.WeightsResponse(updated=False, model_version=version)
        else:
            return learner_pb2.WeightsResponse(updated=True, model_version=version, tensors=weights)
        


def training_loop(agent:DQNAgent,
                  servicer: LearnerService,
                  buffer: ThreadSafeReplayBuffer,
                  stop_event: threading.Event,
                  batch_size: int = 64,
                  min_buffer_size: int = 1000,
                  publish_frequency: int = 10):
    step = 0
    past_losses = deque(maxlen=20)
    while not stop_event.is_set() and len(buffer) < min_buffer_size:
        time.sleep(0.1)
    while not stop_event.is_set():
        batch = buffer.sample(batch_size)
        loss = agent.update(batch)
        past_losses.append(loss)
        step += 1
        
        if step % publish_frequency == 0:
            proto = state_dict_to_proto(agent.q_net.state_dict())
            servicer.update_weights(proto)
        
        if step % 100 == 0:
            print(f"""
                  ===== Step {step} =====
                - Model version: {servicer.version}
                - Buffer length: {len(buffer)}
                - Loss: {sum(past_losses) / len(past_losses):.4f}
                  """)

def main():
    buffer = ThreadSafeReplayBuffer(100000)
    agent = DQNAgent()
    initial_weights = state_dict_to_proto(agent.q_net.state_dict())
    servicer = LearnerService(buffer, initial_weights)
    
    stop_event = threading.Event()
    thread = threading.Thread(target=training_loop, args=[agent, servicer, buffer, stop_event], daemon=True)
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:50051")
    print("Learner gRPC server started on [::]:50051") 
    server.start()
    thread.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        stop_event.set()
        thread.join(timeout=2)
        server.stop(grace=2)

if __name__ == "__main__":
    main()