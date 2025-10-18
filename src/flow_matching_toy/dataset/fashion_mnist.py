from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
