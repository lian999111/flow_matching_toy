import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching_toy.model.time_encoder import get_time_embedding


class FCLayer(nn.Module):
    """A full fully connected layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        return F.relu(self.linear(x))


class ConditionalMLP(nn.Module):
    """A model that outputs a vector with same dimension as the input."""

    def __init__(self, in_channels: int = 2, channels: int = 128, time_emb_channels: int = 128, layers: int = 3):
        super().__init__()
        self.time_emb_channels = time_emb_channels

        self.in_projection = nn.Linear(in_features=in_channels, out_features=channels)
        self.time_projection = nn.Linear(in_features=time_emb_channels, out_features=channels)
        self.layers = nn.Sequential(*[FCLayer(channels, channels) for _ in range(layers)])
        self.out_projection = nn.Linear(in_features=channels, out_features=in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = F.relu(self.in_projection(x))
        t_emb = get_time_embedding(self.time_emb_channels, t)
        t_emb = self.time_projection(t_emb)
        x = self.layers(x + t_emb)
        x = self.out_projection(x)
        return x
