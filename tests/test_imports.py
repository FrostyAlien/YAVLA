"""Smoke test: verify all key dependencies are importable."""

import pytest


def test_torch():
    import torch

    assert torch.__version__
    print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | MPS: {torch.backends.mps.is_available()}")


def test_torchvision():
    import torchvision

    assert torchvision.__version__


def test_torchcodec():
    import torchcodec

    assert torchcodec.__version__


def test_transformers():
    import transformers

    assert transformers.__version__


def test_peft():
    import peft

    assert peft.__version__


def test_accelerate():
    import accelerate

    assert accelerate.__version__


def test_einops():
    import einops

    assert einops.__version__


def test_safetensors():
    import safetensors

    assert safetensors.__version__


def test_wandb():
    import wandb

    assert wandb.__version__


def test_datasets():
    import datasets

    assert datasets.__version__


def test_tyro():
    import tyro

    assert tyro.__version__


def test_lerobot():
    import lerobot

    assert lerobot.__version__


def test_numpy():
    import numpy

    assert numpy.__version__


def test_scipy():
    import scipy

    assert scipy.__version__


def test_rich():
    from importlib.metadata import version

    assert version("rich")


def test_tqdm():
    import tqdm

    assert tqdm.__version__


