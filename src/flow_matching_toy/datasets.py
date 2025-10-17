from pathlib import Path

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def sample_spiral_distribution(num_samples: int, num_turns: float = 2, noise: float = 0.5):
    """
    Samples data points from a 2D spiral distribution.

    Args:
        num_samples (int): The number of data points to generate.
        num_turns (float): The number of times the spiral winds around the center.
        noise (float): The standard deviation of the Gaussian noise added to the points,
                       controlling the "thickness" of the spiral.

    Returns:
        np.ndarray: A NumPy array of shape (num_samples, 2) with the sampled (x, y) coordinates.
    """
    # 1. Generate the base angle (theta)
    # Using sqrt(rand) ensures the point density is uniform across the 2D area
    theta = np.sqrt(np.random.rand(num_samples)) * num_turns * 2 * np.pi

    # 2. Define the spiral's "mean" path
    # The radius is proportional to the angle
    radius = theta

    # 3. Convert the ideal path to Cartesian coordinates
    ideal_x = radius * np.cos(theta)
    ideal_y = radius * np.sin(theta)

    # 4. Add random Gaussian noise to create the distribution
    # This creates a "cloud" of points around the ideal spiral line
    sampled_x = ideal_x + np.random.randn(num_samples) * noise
    sampled_y = ideal_y + np.random.randn(num_samples) * noise

    # Combine x and y into a single array
    points = np.vstack((sampled_x, sampled_y)).T

    return points


def get_fashion_mnist_dataloaders(batch_size: int, image_size: int = 32, num_workers=4):
    """
    Downloads the Fashion-MNIST dataset and prepares DataLoaders.

    Args:
        batch_size (int): The number of samples per batch.
        image_size (int): The target size to resize images to.

    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    # TODO: Make this more robust
    data_root = Path(__file__).resolve().parent.parent / "data"

    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
