import pytest
import torch

from flow_matching_toy.model.models import ConditionalMLP, FiLMLayer, ConditionalResBlock, ConditionalUNet


class TestConditionalMLP:
    def test_output_shape(self):
        model = ConditionalMLP(in_channels=2, channels=16, time_emb_channels=8, layers=1)
        batch_size = 4
        x = torch.rand((batch_size, 2))
        t = torch.rand((batch_size,))

        y = model(x, t)

        assert y.shape == x.shape


# ---------------------------- U-Net ----------------------------
# --- Test Fixtures: Reusable test data ---


@pytest.fixture
def test_data():
    """Provides a dictionary of common tensor shapes and data."""
    return {
        "batch_size": 4,
        "height": 32,
        "width": 32,
        "cond_channels": 48,
        "feature": torch.randn(4, 64, 32, 32),
        "condition": torch.randn(4, 48),
    }


class TestFiLM:
    def test_output_shape(self, test_data):
        """Tests if FiLMLayer output shape matches the input feature shape."""
        feat_channels = test_data["feature"].shape[1]
        layer = FiLMLayer(cond_channels=test_data["cond_channels"], feat_channels=feat_channels)

        output = layer(test_data["feature"], test_data["condition"])

        assert output.shape == test_data["feature"].shape, "Output shape must match feature shape"

    def test_modulation_effect(self, test_data):
        """Tests that the FiLM layer actually modifies the input feature."""
        feat_channels = test_data["feature"].shape[1]
        layer = FiLMLayer(cond_channels=test_data["cond_channels"], feat_channels=feat_channels)

        output = layer(test_data["feature"], test_data["condition"])

        assert not torch.equal(output, test_data["feature"]), "Output should not be identical to input"


# --- 2. Tests for ConditionalResBlock ---


@pytest.mark.parametrize(
    "in_channels, out_channels",
    [
        (32, 64),  # Test case where channels change
        (64, 64),  # Test case where channels stay the same
    ],
)
class TestResBlock:
    def test_shape(self, test_data, in_channels, out_channels):
        """Tests output shape for both same and different channel sizes."""
        block = ConditionalResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=test_data["cond_channels"],
        )
        x = torch.randn(test_data["batch_size"], in_channels, test_data["height"], test_data["width"])

        output = block(x, test_data["condition"])

        expected_shape = (test_data["batch_size"], out_channels, test_data["height"], test_data["width"])
        assert output.shape == expected_shape, "Output shape is incorrect"

    def test_gradient_flow(self, test_data, in_channels, out_channels):
        """Tests that gradients flow through the ConditionalResBlock."""
        block = ConditionalResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=test_data["cond_channels"],
        )
        x = torch.randn(test_data["batch_size"], in_channels, test_data["height"], test_data["width"])

        output = block(x, test_data["condition"])
        loss = output.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"Gradient is None for parameter {name}"
            assert torch.sum(param.grad**2) > 0, f"Gradient is zero for parameter {name}"


# --- 3. Tests for ConditionalUNet (End-to-End) ---


class TestConditionalUNet:
    def test_output_shape(self, test_data):
        """Tests that the U-Net output shape matches the input image shape."""
        io_channels = 3
        model = ConditionalUNet(io_channels=io_channels, cond_channels=test_data["cond_channels"])
        x = torch.randn(test_data["batch_size"], io_channels, test_data["height"], test_data["width"])

        output = model(x, test_data["condition"])

        assert output.shape == x.shape, "Final output shape must match input image shape"

    def test_end_to_end_gradient_flow(self, test_data):
        """Tests gradient flow through the entire U-Net."""
        io_channels = 3
        model = ConditionalUNet(io_channels=io_channels, cond_channels=test_data["cond_channels"])
        x = torch.randn(test_data["batch_size"], io_channels, test_data["height"], test_data["width"])

        output = model(x, test_data["condition"])
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for parameter {name}"
                assert torch.sum(param.grad**2) > 0, f"Gradient is zero for parameter {name}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility(self, test_data):
        """Tests if the U-Net can be moved to and run on a CUDA device."""
        device = "cuda"
        io_channels = 3
        model = ConditionalUNet(io_channels=io_channels, cond_channels=test_data["cond_channels"]).to(device)
        x = torch.randn(test_data["batch_size"], io_channels, test_data["height"], test_data["width"], device=device)
        condition = test_data["condition"].to(device)

        try:
            output = model(x, condition)
            assert output.device.type == device, "Output tensor is not on the correct device"
        except Exception as e:
            pytest.fail(f"Model failed on CUDA device with error: {e}")
