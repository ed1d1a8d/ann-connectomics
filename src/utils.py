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
