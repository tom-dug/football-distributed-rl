from torch import Tensor, nn, randn, isnan

class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 115, action_dim: int = 19, hidden_dim: int = 256):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.stack(x)
    



if __name__ == "__main__":
    """
    Testing implementation
    """
    net = QNetwork()

    output1 = net.forward(randn((1, 115)))
    output2 = net.forward(randn((64, 115)))


    print(f"Output 1: {output1.shape}")
    print(f"Output 2: {output2.shape}")

    print(f"Output 1: {isnan(output1).any()}")
    print(f"Output 2: {isnan(output2).any()}")