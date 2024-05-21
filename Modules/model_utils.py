from collections import OrderedDict
from torch import nn
from functools import partial


def construct_layers(layer_conf: list[str], input_size, hidden_size, layer_idx) -> list:
    """Constructs a layer topology."""
    layers = []
    for layer_spec in layer_conf:
        layer = eval(layer_spec)
        layers.append((f"layer{layer_idx}", layer))
        layer_idx += 1
    return layers, layer_idx
