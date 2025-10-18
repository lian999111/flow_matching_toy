import math

import torch


def get_time_embedding(embedding_dim: int, time: torch.Tensor, max_position: int = 10000):
    """Creates sinusoidal time embeddings.

    This function maps a batch of continuous time values (t) ranging from [0, 1]
    into high-dimensional vectors using a sinusoidal embedding technique. This is
    commonly used in diffusion models to condition the model on the noise level.

    Args:
        embedding_dim: The dimensionality of the output embedding vector.
        time: A 1D tensor of shape (B,) containing normalized time steps,
              where B is the batch size.
        max_position: The maximum value for the scaled time, used to define
                      the frequency range of the sinusoids. Defaults to 10000.

    Returns:
        A 2D tensor of shape (B, embedding_dim) containing the time embeddings.
    """
    # Scale up time from [0, 1] to [0, max_position] since position embedding is designed to work with large position possibilities
    time = time * max_position
    half_dim = embedding_dim // 2
    freq = (
        torch.arange(half_dim, device=time.device, dtype=torch.float) * -(math.log(max_position) / (half_dim - 1))
    ).exp()
    input = time[:, None] * freq[None, :]  # expand axes for broadcasting
    embedding = torch.concat([torch.sin(input), torch.cos(input)], dim=1)  # (B, embedding_dim)

    # Handle odd embedding dimensions by padding
    if embedding_dim % 2 != 0:
        embedding = torch.nn.functional.pad(embedding, (0, 1))

    return embedding
