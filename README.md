## Flow Matching Toy

This repo is inspired by [this tutorial repo](https://github.com/dome272/Flow-Matching) and contains my exploration of [flow matching for generative models](https://arxiv.org/abs/2210.02747), a technique utilized by state-of-the-art generative models like Stable Diffusion 3.
Currently, a simple MLP model is trained to learn a vector flow that pushes data samples from a standard Gaussian distribution to a spiral-shaped distribution.
While it is simple, it captures largely the idea behind flow-matching generative models and the very same concept can be applied directly to much more complicated generative tasks and models. Next steps would be to train a simple image generation model and try out classifier-free guidance.

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
* Run the script `flow_matching_toy/main.py`. It is recommended to run it in vscode's interactive mode to get a Jupyter-Notebook-like experience.

### Run unit tests
Pytest has been config in `pyproject.toml` and will be installed with the `--all-extras` arg when running `uv sync`. After that, simple run:
```bash
pytest
```