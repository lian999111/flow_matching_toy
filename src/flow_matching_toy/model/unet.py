import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------------- U-Net ----------------------------


class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to a feature map.

    This layer generates a scale (`gamma`) and shift (`beta`) parameter for each
    feature channel based on a conditioning vector, and then applies an affine
    transformation to the input feature map.
    """

    def __init__(self, cond_channels: int, feat_channels: int):
        """
        Initializes the FiLMLayer.

        Args:
            cond_channels: The number of channels in the conditioning vector.
            feat_channels: The number of channels in the feature map to be modulated.
        """
        super().__init__()
        # Need two sets: gemma and beta
        self.linear = nn.Linear(in_features=cond_channels, out_features=2 * feat_channels)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor):
        """
        Apply the FiLM transformation to the input feature.

        Args:
            feature: The input feature map of shape (B, C_feat, H, W).
            condition: The conditioning vector of shape (B, C_cond).

        Returns:
            torch.Tensor: The modulated feature map, with the same shape as the input feature.
        """
        gemma_and_beta = F.gelu(self.linear(condition))
        gemma, beta = torch.chunk(gemma_and_beta, 2, dim=1)
        # Unsqueeze H, W dimension for broadcasting
        return gemma[:, :, None, None] * feature + beta[:, :, None, None]


class ConditionalResBlock(nn.Module):
    """
    A residual block that incorporates conditioning information via a FiLM layer.

    This block processes an input tensor through two convolutional layers, with
    FiLM-based conditioning applied after the first normalization. It includes a
    residual connection from the input to the output.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_channels: int):
        """
        Initializes the ConditionalResBlock.

        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels in the output tensor.
            cond_channels: The number of channels in the conditioning vector.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.film = FiLMLayer(cond_channels=cond_channels, feat_channels=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Forward pass for the ConditionalResBlock.

        Args:
            x: The input tensor of shape (B, C_in, H, W).
            condition: The conditioning vector of shape (B, C_cond).

        Returns:
            torch.Tensor: The output tensor of shape (B, C_out, H, W).
        """
        h = self.norm1(self.conv1(x))
        h = self.film(h, condition)
        h = F.gelu(h)

        h = self.norm2(self.conv2(h))
        h = F.gelu(h)
        return h + self.residual_conv(x)


class ConditionalUNet(nn.Module):
    """
    A U-Net architecture that accepts a conditioning vector at each residual block.

    The U-Net consists of a down-sampling path (encoder), a bottleneck, and an
    up-sampling path (decoder) with skip connections. The conditioning vector
    is passed to each ConditionalResBlock in the network.
    """

    def __init__(self, io_channels: int, cond_channels: int):
        """
        Initializes the ConditionalUNet.

        Args:
            io_channels: The number of channels for the input and output images.
            cond_channels: The number of channels in the conditioning vector.
        """
        super().__init__()
        self.encoder_block1 = ConditionalResBlock(in_channels=io_channels, out_channels=32, cond_channels=cond_channels)
        self.encoder_block2 = ConditionalResBlock(in_channels=32, out_channels=64, cond_channels=cond_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ConditionalResBlock(in_channels=64, out_channels=128, cond_channels=cond_channels)

        self.decoder_block1 = ConditionalResBlock(in_channels=128 + 64, out_channels=64, cond_channels=cond_channels)
        self.decoder_block2 = ConditionalResBlock(in_channels=64 + 32, out_channels=32, cond_channels=cond_channels)
        self.up_sample = nn.Upsample(scale_factor=2)

        self.output_conv = nn.Conv2d(in_channels=32, out_channels=io_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Forward pass for the ConditionalUNet.

        Args:
            x: The input image tensor of shape (B, C_io, H, W).
            condition: The conditioning vector of shape (B, C_cond).

        Returns:
            torch.Tensor: The output tensor, with the same shape as the input x.
        """
        # Encoder
        skip1 = self.encoder_block1(x, condition)
        h = self.pool(skip1)
        skip2 = self.encoder_block2(h, condition)
        h = self.pool(skip2)

        h = self.bottleneck(h, condition)

        # Decoder
        h = self.up_sample(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.decoder_block1(h, condition)
        h = self.up_sample(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.decoder_block2(h, condition)

        return self.output_conv(h)
