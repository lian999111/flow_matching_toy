import torch

from flow_matching_toy.models import MLP


class TestMLP:
    def test_output_shape(self):
        model = MLP(in_channels=2, channels=16, time_emb_channels=8, layers=1)
        batch_size = 4
        x = torch.rand((batch_size, 2))
        t = torch.rand((batch_size,))

        y = model(x, t)

        assert y.shape == x.shape
