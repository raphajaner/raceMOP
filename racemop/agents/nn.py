import torch
import torch.nn as nn
from racemop.agents.nn_utils import layer_init


class LidarNet(nn.Module):
    """Network for processing lidar data."""

    def __init__(self, config, in_channel=1):
        super().__init__()

        self.register_buffer('n_datapoints', torch.tensor(1080))
        self.register_buffer('n_channels', torch.tensor(config.env.frame_cat.n))

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(config.rl.hidden_out_dim),
            nn.ReLU(),
        )

    def reshape(self, data):
        """Reshapes the data to the correct shape for the network."""
        return data.view(-1, self.n_channels, self.n_datapoints)

    def forward(self, input):
        """Forward pass of the network."""
        return self.encoder_linear(self.encoder_conv(input))


class ActorMeanNet(nn.Sequential):
    def __init__(self, config, in_features):
        """Actor model for mean values of action distribution."""
        model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 2), std=0.01),
        )
        super().__init__(model)


class CriticNet(nn.Sequential):
    def __init__(self, config, in_features):
        """Actor model for mean values of action distribution."""
        model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        super().__init__(model)
