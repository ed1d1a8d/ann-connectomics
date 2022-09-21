import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def ceil_div(a: int, b: int) -> int:
    assert a >= 0 and b > 0
    return (a + b - 1) // b


def plot_loghist(vals, weights, bins: int):
    logbins = np.logspace(np.log10(min(vals)), np.log10(max(vals)), bins)
    plt.hist(vals, weights=weights, bins=logbins)  # type: ignore
    plt.xscale("log")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_children(model: torch.nn.Module) -> list[nn.Module]:
    """Adapted from https://stackoverflow.com/a/65112132/1337463."""
    children = list(model.children())

    # if model has no children; model is last child! :O
    if children == []:
        return [model]

    flat_children = []
    for child in children:
        flat_children.extend(get_flat_children(child))
    return flat_children


def get_flat_nodes(model: torch.nn.Module) -> list[nn.Module]:
    """Adapted from https://stackoverflow.com/a/65112132/1337463."""
    children = list(model.children())

    if children == []:
        return [model]

    flat_nodes = [model]
    for child in children:
        flat_nodes.extend(get_flat_nodes(child))
    return flat_nodes
