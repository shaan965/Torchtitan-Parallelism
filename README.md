# TorchTitan Parallelism Memory Explorer

This notebook demonstrates how different distributed training strategies affect per-GPU memory usage for large Llama 3-style models, without requiring real GPUs.

It uses PyTorch FakeTensorMode and TorchTitan utilities to simulate model parallelism and estimate whether a chosen configuration fits within GPU memory.

## What It Does

The notebook lets users explore memory usage for:

- Tensor Parallelism (TP)
- Context Parallelism (CP)
- Fully Sharded Data Parallelism (FSDP)
- Hybrid Sharded Data Parallelism (HSDP)
- Batch size and sequence length changes

It reports whether a configuration:

- `FITS`
- is `NEAR OOM`
- or results in `OOM`

The estimate includes parameters, gradients, activations, communication buffers, and AdamW optimizer state.

## Scenarios

The notebook includes three predefined scenarios:

| Scenario | Model | World Size | GPU Memory |
|---|---:|---:|---:|
| 1 | Llama3 8B | 8 GPUs | 40 GiB |
| 2 | Llama3 70B | 128 GPUs | 80 GiB |
| 3 | Llama3 405B | 1024 GPUs | 80 GiB |

## Open in Colab

Open the notebook here:

[parallelism_explorer.ipynb](https://colab.research.google.com/github/fegin/titan-demo/blob/main/notebooks/parallelism_explorer.ipynb)

## Setup

The notebook installs the required dependencies automatically:

```python
pip install --pre torch
pip install git+https://github.com/pytorch/torchtitan.git
pip install git+https://github.com/fegin/titan-demo.git