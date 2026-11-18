import torch
import learner_pb2
import numpy as np

def state_dict_to_proto(state_dict: dict[str, torch.Tensor]) -> list[learner_pb2.TensorData]:
    tensor_data_list = []
    for name, tensor in state_dict.items():
        tensor_data = learner_pb2.TensorData(
            name=name,
            shape=list(tensor.shape),
            data=tensor.detach().cpu().contiguous().numpy().tobytes()
        )
        tensor_data_list.append(tensor_data)
    return tensor_data_list


def proto_to_state_dict(tensor_data: list[learner_pb2.TensorData]) -> dict[str, torch.Tensor]:
    new_dict = {}
    for td in tensor_data:
        raw_bytes = td.data
        np_tensor = np.frombuffer(raw_bytes, dtype=np.float32).reshape(td.shape).copy()
        tensor = torch.from_numpy(np_tensor)
        new_dict[td.name] = tensor
    return new_dict


