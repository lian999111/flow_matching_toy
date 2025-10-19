import torch

from flow_matching_toy.model.time_encoder import get_time_embedding


def test_output_shape_even_dim():
    """Tests if the output shape is correct for an even embedding dimension."""
    batch_size = 4
    embed_dim = 256
    time = torch.rand(batch_size)

    embedding = get_time_embedding(embed_dim, time)

    assert embedding.shape == (batch_size, embed_dim)


def test_output_shape_odd_dim():
    """Tests if the output shape is correct for an odd embedding dimension (checks padding)."""
    batch_size = 4
    embed_dim = 257
    time = torch.rand(batch_size)

    embedding = get_time_embedding(embed_dim, time)

    assert embedding.shape == (batch_size, embed_dim)


def test_known_values_at_time_zero():
    """Tests the output for the known edge case of t=0."""
    batch_size = 2
    embed_dim = 128
    time = torch.zeros(batch_size)  # t = 0

    embedding = get_time_embedding(embed_dim, time)

    # Expected pattern for t=0: sin(0)=0, cos(0)=1 -> [0, 1, 0, 1, ...]
    half_dim = embed_dim // 2
    expected_row = torch.cat([torch.zeros(half_dim), torch.ones(half_dim)])
    expected_embedding = expected_row.unsqueeze(0).expand((batch_size, -1))

    # Use torch.allclose for safe floating-point comparison
    assert torch.allclose(embedding, expected_embedding)


def test_known_values_at_time_one():
    """Tests the output for the known edge case of t=1."""
    batch_size = 2
    embed_dim = 4
    time = torch.ones(batch_size)

    embedding = get_time_embedding(embed_dim, time, max_position=10000)

    expected_row = torch.tensor([-0.3056, 0.8415, -0.9522, 0.5403])
    expected_embedding = expected_row.unsqueeze(0).expand((batch_size, -1))

    # Use torch.allclose for safe floating-point comparison
    assert torch.allclose(embedding, expected_embedding, atol=1e-3)
