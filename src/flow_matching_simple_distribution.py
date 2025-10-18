"""
Script to train and inference a flow-matching model to,
given a sample from a Gaussian distribution,
generate a data sample in a spiral distribution.
"""

# NOTE: This script uses vscode's interactive notebook feature that allows for cell-by-cell execution.

# %%
import torch
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

from flow_matching_toy.dataset.spiral_distribution import sample_spiral_distribution
from flow_matching_toy.model.models import ConditionalMLP

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
model = ConditionalMLP(in_channels=2, channels=256, time_emb_channels=128, layers=5).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

NUM_TRAIN_STEPS = 10000
BATCH_SIZE = 128

losses = []
for step in tqdm(range(NUM_TRAIN_STEPS)):
    target_dist_samples = sample_spiral_distribution(num_samples=BATCH_SIZE)
    target_dist_samples = torch.as_tensor(target_dist_samples, device=device, dtype=torch.float)
    random_noise = torch.randn_like(target_dist_samples)

    time = torch.rand((BATCH_SIZE,), device=device)
    time_unsqueezed = time.unsqueeze(1)
    x = time_unsqueezed * target_dist_samples + (1 - time_unsqueezed) * random_noise
    target_flow = target_dist_samples - random_noise

    pred_flow = model(x, time)

    loss = F.mse_loss(pred_flow, target_flow)
    losses.append(loss.detach().cpu().numpy())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# %%
plt.plot(losses)
plt.title("Loss")

# %% Inference
with torch.no_grad():
    model.eval()
    NUM_SAMPLES = 1000
    NUM_STEPS = 1000
    x = torch.randn((NUM_SAMPLES, 2), device=device)

    for t in torch.linspace(0, 1, NUM_STEPS, device=device):
        t = t.expand((NUM_SAMPLES,))
        flow = model(x, t)
        x += 1 / NUM_STEPS * flow

    x = x.cpu().numpy()
    plt.scatter(x[:, 0], x[:, 1])

# %%
