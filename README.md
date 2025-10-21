## Flow Matching Toy

This repo is inspired by [this tutorial repo](https://github.com/dome272/Flow-Matching) and contains my exploration of [flow matching for generative models](https://arxiv.org/abs/2210.02747), a technique utilized by state-of-the-art generative models like Stable Diffusion 3.

Currently, two experiments are implemented:
* A simple MLP model is trained to learn a vector flow that pushes data samples from a standard Gaussian distribution to a spiral-shaped distribution.
* A conditional UNet for image generation conditioned on class input using classifier-free guidance. It's currently trained on Fashion-MNIST for simplicity. Next step would be to train it on more interesting datasets like CIFRA-10.

## Quick Start
This project is manage by `uv`, a modern python package and project manager.

* Install `uv` following [this](https://docs.astral.sh/uv/getting-started/installation/) if not done yet.
* Create a virtual env
```bash
uv venv
```
* Set up environmetn using `uv.lock` file
```bash
uv sync --all-extras
```
* Install this repo in the editable mode
```bash
uv install -e .
```
* Run the script:
    * `flow_matching_toy/flow_matching_simple_distribution.py` to see how an MLP can be trained to map a simple Gaussian distribution to a spiral distribution. It is recommended to run it in vscode's interactive mode to get a Jupyter-Notebook-like experience.
    * `flow_matching_toy/flow_matching_image_gen.py` to train a conditional UNet to generate images given a class condition.
      A notebook `flow_matching_toy/gen_image_using_flow_matcher.ipynb` can be used to visually evaluate the generated image.

### Run unit tests
Pytest has been config in `pyproject.toml` and will be installed with the `--all-extras` arg when running `uv sync`. After that, simple run:
```bash
pytest
```