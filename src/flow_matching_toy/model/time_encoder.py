import math

import torch
import torch.nn as nn


class SinusoidalTimeEncoder(nn.Module):
    """
    Time encoder for flow matching that uses sinusoidal embeddings.

    This module maps a batch of continuous time values (t) ranging from [0, 1]
    into high-dimensional vectors. It's a standard component in diffusion and
    flow-matching models.
    """

    def __init__(self, embedding_dim: int, max_position=10000):
        """
        Args:
            embedding_dim: The dimensionality of the output embedding vector.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_position = max_position

        # Precompute and cache freq tensor
        half_dim = embedding_dim // 2
        freq = (torch.arange(half_dim, dtype=torch.float) * -(math.log(max_position) / (half_dim - 1))).exp()
        # This ensures pytorch correctly handles `self.freq` during device placement and state saving/loading
        self.register_buffer("freq", freq)

    def forward(self, time: torch.Tensor):
        """
        Get sinusoidal time embedding.

        Args:
            time: Time in the range of [0, 1].
        """
        # NOTE: We don't assert time is in [0, 1] since this might break when exporting the model to ONNX
        # Instead, make sure the time is valid outside the mode whenever possible

        # Scale up time from [0, 1] to [0, max_position] since position embedding is designed to work with large position possibilities
        time = time * self.max_position

        input = time[:, None] * self.freq[None, :]  # expand axes for broadcasting
        embedding = torch.concat([torch.sin(input), torch.cos(input)], dim=1)  # (B, embedding_dim)

        # Handle odd embedding dimensions by padding
        if self.embedding_dim % 2 != 0:
            embedding = torch.nn.functional.pad(embedding, (0, 1))

        return embedding
