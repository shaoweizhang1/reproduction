"""Optional GPU placement for the neural half of a model.

Works around a bug in deepproblog.network.Network (https://github.com/ML-KULeuven/deepproblog): its batching
branch calls inputs.cuda(device=...) without assigning the result, so the
inputs stay on the CPU and the forward pass dies on a device mismatch.
"""

import os

import torch

from deepproblog.network import Network

def resolve(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return device

def move(value, device):
    if isinstance(value, (list, tuple)):
        return [tensor.to(device) for tensor in value]
    return value.to(device)

class GpuNetwork(Network):

    def __call__(self, to_evaluate):
        if not (self.is_cuda and self.batching):
            return super().__call__(to_evaluate)
        stacked_inputs = []
        for group in zip(*(self.function(*e) for e in to_evaluate)):
            try:
                group = torch.stack(group).cuda(device=self.device)
            except TypeError:
                group = list(group)
            stacked_inputs.append(group)
        return self.network_module(*stacked_inputs)
