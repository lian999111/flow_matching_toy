from .mlp import ConditionalMLP
from .unet import ConditionalUNet

MODEL_REGISTRY = {
    "conditional_mlp": ConditionalMLP,
    "conditional_unet": ConditionalUNet,
}
