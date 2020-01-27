import torch.nn as nn


def get_conv_output_size(input_size, layers):
    size = input_size
    for layer in layers:
        if type(layer) in [nn.Conv2d, nn.MaxPool2d]:
            size = get_layer_output_size(
                size, layer.kernel_size, layer.padding,
                layer.stride, layer.dilation
            )
    return size


def get_layer_output_size(input_size, kernel_size, padding, stride, dilation):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return (
        (
            input_size[0] + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1,
        (
            input_size[1] + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1,
    )
