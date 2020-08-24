import torch
import torch.nn as nn


class ModuleCompose(nn.Module):
    def __init__(self, *modules_and_functions):
        super().__init__()
        self.modules_and_functions = modules_and_functions

        self.module_list = nn.ModuleList([
            module_or_function
            for module_or_function in self.modules_and_functions
            if isinstance(module_or_function, nn.Module)
        ] + [
            module_or_function[0]
            for module_or_function in self.modules_and_functions
            if (
                type(module_or_function) is tuple and
                isinstance(module_or_function[0], nn.Module)
            )
        ])

        self.parameter_list = nn.ParameterList([
            parameter
            for parameter in self.modules_and_functions
            if isinstance(parameter, nn.Parameter)
        ] + [
            parameter_or_tuple[0]
            for parameter_or_tuple in self.modules_and_functions
            if (
                type(parameter_or_tuple) is tuple and
                isinstance(parameter_or_tuple[0], nn.Parameter)
            )
        ])


    def forward(self, *x):
        for module_or_function in self.modules_and_functions:
            if type(module_or_function) is tuple:
                module, fn = module_or_function
                if type(x) is tuple:
                    x = fn(module, *x)
                else:
                    x = fn(module, x)
            else:
                if type(x) is tuple:
                    x = module_or_function(*x)
                else:
                    x = module_or_function(x)
        return x

    @torch.no_grad()
    def debug(self, x):
        for index, module_or_function in enumerate(self.modules_and_functions):
            if type(module_or_function) is tuple:
                module, fn = module_or_function

                if isinstance(module, nn.Module):
                    n_parameters = sum(
                        [p.shape.numel() for p in module.parameters()]
                    )
                    n_parameters_postfix = f' n_parameters: {n_parameters}'
                else:
                    n_parameters_postfix = ''

                print_intermediate(index, x, n_parameters_postfix)
                if type(x) is tuple:
                    x = fn(module, *x)
                else:
                    x = fn(module, x)
            else:
                if isinstance(module_or_function, nn.Module):
                    n_parameters = sum([
                        p.shape.numel()
                        for p in module_or_function.parameters()
                    ])
                    n_parameters_postfix = f' n_parameters: {n_parameters}'
                else:
                    n_parameters_postfix = ''

                print_intermediate(index, x, n_parameters_postfix)
                if type(x) is tuple:
                    x = module_or_function(*x)
                else:
                    x = module_or_function(x)
        return x


def print_intermediate(index, x, postfix):
    if type(x) is tuple:
        if hasattr(x[0], 'shape'):
            representation = f'shape: {[y.shape for y in x]}'
        else:
            representation = f'type: {[type(y) for y in x]}'
    else:
        if hasattr(x, 'shape'):
            representation = f'shape: {x.shape}'
        else:
            representation = f'type: {type(x)}'
    print(f'index: {index}, {representation}' + postfix)


def test_module_compose():
    import numpy as np

    class Example:
        def __init__(self, data):
            self.data = data

    model = ModuleCompose(
        lambda examples: torch.stack([
            torch.from_numpy(example.data).float() for example in examples
        ]),
        nn.Conv2d(3, 32, 5),
        lambda x: x.mean(dim=(-1, -2)),
        lambda x: x.view(x.size(0), -1),
        nn.Linear(32, 1),
    )

    batch = [Example(np.random.randn(3, 32, 32)) for i in range(8)]
    model.debug(batch)
