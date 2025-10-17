from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Module

from flow_matching_toy.time_embedding import get_time_embedding
from flow_matching_toy.datasets import get_fashion_mnist_dataloaders


class FlowMatchingModel(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        condition_encoder: Module,
        config: dict,
    ):
        super().__init__()
        self.model = model
        self.condition_encoder = condition_encoder
        self.config = config

    def forward(self, x_t, condition, time):
        cond_emb = self.condition_encoder(condition)
        # TODO: Think of a better way to config this
        time_emb = get_time_embedding(embedding_dim=self.config["cond_emb_dim"] / 2, time=time)
        # Include time embedding as part of condition embedding
        cond_emb = torch.cat([time_emb, cond_emb], dim=1)
        pred_velocity = self.model(x_t, cond_emb)
        return pred_velocity

    def training_step(self, batch: Any, batch_idx: int):
        """Train one step with each batch containing target and condition."""
        x_target, condition = batch
        batch_size = x_target.shape[0]

        random_noise = torch.randn_like(x_target)
        time = torch.rand(batch_size, device=x_target.device)
        x_t = (1 - time.view(-1, 1, 1, 1)) * random_noise + time.view(-1, 1, 1, 1) * x_target

        # Forward pass
        pred_velocity = self(x_t, condition, time)

        true_velocity = x_target - random_noise
        loss = F.mse_loss(true_velocity, pred_velocity)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"])
        return optim

def train_flow_matching():
    """Train flow matching model."""
    from flow_matching_toy.models import ConditionalUNet
    from torch.nn import Embedding

    model = ConditionalUNet(io_channels=1, cond_channels=32)
    # Use a simple embedding layer as the encoder
    cond_encoder = Embedding(num_embeddings=11, embedding_dim=16)

    train_loader, val_loader = get_fashion_mnist_dataloaders(batch_size=16, image_size=32, num_workers=4)

    config = {
        "cond_emb_dim": 32,
        "learning_rate": 1e-3,
    }

    flow_match_model = FlowMatchingModel(
        model=model,
        condition_encoder=cond_encoder,
        config=config,
    )

    trainer = pl.Trainer(max_epochs=1)
    # trainer = pl.Trainer(overfit_batches=1, max_epochs=10, log_every_n_steps=1)
    trainer.fit(flow_match_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train_flow_matching()