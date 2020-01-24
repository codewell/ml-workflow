import torch.nn as nn
from .get_layer_output_size import get_layer_output_size


def get_conv_output_size(input_size, layers):
    size = input_size
    for layer in layers:
        if type(layer) in [nn.Conv2d, nn.MaxPool2d]:
            size = get_layer_output_size(
                size, layer.kernel_size, layer.padding,
                layer.stride, layer.dilation
            )
    return size
