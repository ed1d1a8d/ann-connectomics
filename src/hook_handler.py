"""
Context manager for pytorch hooks.
"""

import contextlib
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class HookHandler(contextlib.AbstractContextManager):
    def __init__(self):
        self.activations = {}
        self.hook_handles: list[RemovableHandle] = []

    def reset(self):
        for h in self.hook_handles:
            h.remove()

        self.activations = {}
        self.hook_handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()

    def add_hook(
        self,
        mod: nn.Module,
        fn: Callable[[nn.Module, Any, torch.Tensor], Any],
    ):
        self.hook_handles.append(mod.register_forward_hook(fn))
