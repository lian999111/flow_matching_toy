# %%
from flow_matching_toy.data_sampler import sample_spiral_distribution

import matplotlib.pyplot as plt

# %%
points = sample_spiral_distribution(num_samples=1000, num_turns=2, noise=0.5)

plt.scatter(points[:, 0], points[:, 1])
