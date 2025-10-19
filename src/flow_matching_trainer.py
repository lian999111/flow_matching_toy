# %%
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from flow_matching_toy.model import MODEL_REGISTRY
from flow_matching_toy.model.time_encoder import SinusoidalTimeEncoder
from flow_matching_toy.dataset.fashion_mnist import get_fashion_mnist_dataloaders


class FlowMatchingModel(pl.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        cond_encoder_cfg: Dict[str, Any],
        time_encoder_cfg: Dict[str, Any],
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            model_cfg (dict): Configuration dictionary for the main model.
                              Must contain a 'name' key mapping to MODEL_REGISTRY
                              and a 'params' key with a dict of model hyperparameters.
            cond_encoder_cfg (dict): Configuration dictionary for the condition encoder.
                                     Must contain a 'name' key mapping to
                                     CONDITION_ENCODER_REGISTRY and a 'params' key.
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        # TODO: Make time encoder configurable
        # self.time_encoder_cfg = self.hparams.time_encoder_cfg

        model_name = model_cfg["name"]
        model_params = model_cfg["params"]
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found.")
        self.model = MODEL_REGISTRY[model_name](**model_params)

        # TODO: Enable choosing these encoder models
        self.condition_encoder = torch.nn.Embedding(**cond_encoder_cfg)
        self.time_encoder = SinusoidalTimeEncoder(**time_encoder_cfg)

    def forward(self, x_t, condition, time):
        cond_emb = self.condition_encoder(condition)
        # TODO: Think of a better way to config this
        time_emb = self.time_encoder(time)
        # Include time embedding as part of condition embedding
        cond_emb = torch.cat([time_emb, cond_emb], dim=1)
        pred_velocity = self.model(x_t, cond_emb)
        return pred_velocity

    def step(self, batch: Any):
        """Compute the loss on one batch containing target and condition."""
        x_target, condition = batch
        batch_size = x_target.shape[0]

        random_noise = torch.randn_like(x_target)
        time = torch.rand(batch_size, device=x_target.device)
        x_t = (1 - time.view(-1, 1, 1, 1)) * random_noise + time.view(-1, 1, 1, 1) * x_target

        # Forward pass
        pred_velocity = self(x_t, condition, time)

        true_velocity = x_target - random_noise
        loss = F.mse_loss(true_velocity, pred_velocity)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """Train one step with each batch containing target and condition."""
        # TODO: Add Classifier-Free Guidance
        loss = self.step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Validate one step with each batch containing target and condition."""
        val_loss = self.step(batch)
        self.log("val_loss", val_loss, on_epoch=True)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optim


def train_flow_matching():
    """Train flow matching model."""
    train_loader, val_loader = get_fashion_mnist_dataloaders(batch_size=16, image_size=32, num_workers=4)

    model_cfg = {
        "name": "conditional_unet",
        "params": {
            "io_channels": 1,
            "cond_channels": 32,  # 16 for class condition, 16 for time
        },
    }

    cond_encoder_cfg = {
        "num_embeddings": 11,  # 10 for fashion-mnist and 1 for null (unconditional class)
        "embedding_dim": 16,
    }

    time_encoder_cfg = {
        "embedding_dim": 16,
    }

    assert (
        model_cfg["params"]["cond_channels"] == cond_encoder_cfg["embedding_dim"] + time_encoder_cfg["embedding_dim"]
    ), "Condition embedding dimensions don't match."

    flow_match_model = FlowMatchingModel(
        model_cfg=model_cfg, cond_encoder_cfg=cond_encoder_cfg, time_encoder_cfg=time_encoder_cfg, learning_rate=1e-3
    )

    trainer = pl.Trainer(max_epochs=10)
    # trainer = pl.Trainer(overfit_batches=1, max_epochs=10, log_every_n_steps=1)
    trainer.fit(flow_match_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return flow_match_model


if __name__ == "__main__":
    train_flow_matching()

# # %%
# flow_match_model = FlowMatchingModel.load_from_checkpoint("/home/arry_iang/workspace/playground/flow_matching_toy/src/lightning_logs/version_6/checkpoints/epoch=9-step=37500.ckpt")

# # %%
# flow_match_model.eval()

# import matplotlib.pyplot as plt
# NUM_SAMPLES = 1
# NUM_STEPS = 100
# CONDITION = 5
# x = torch.randn((NUM_SAMPLES, 1, 32, 32), device="cuda")
# for t in torch.linspace(0, 1, NUM_STEPS, device="cuda"):
#     t = t.expand((NUM_SAMPLES,))
#     condition = torch.tensor((CONDITION,), dtype=int, device="cuda")
#     flow = flow_match_model(x, condition, t)
#     x += 1 / NUM_STEPS * flow

# plt.imshow(x.cpu().detach()[0].permute(1, 2, 0))
