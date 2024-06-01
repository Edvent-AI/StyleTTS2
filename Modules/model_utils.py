from collections import OrderedDict

from torch import nn


def construct_layers(layer_conf: list[str]) -> nn.Sequential:
    """Constructs a layer topology."""
    layers = []
    for idx, layer_spec in enumerate(layer_conf):
        layer = eval(layer_spec)
        layers.append((f'layer{idx}', layer))
    layers = nn.Sequential(OrderedDict(layers))
    
    return layers