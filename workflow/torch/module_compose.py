import torch
import torch.nn as nn


class ModuleCompose(nn.Module):
    def __init__(self, *modules_and_functions):
        super().__init__()
        self.modules_and_functions = modules_and_functions

        self.modules = nn.ModuleList([
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

    def debug(self, x):
        for index, module_or_function in enumerate(self.modules_and_functions):
            if type(module_or_function) is tuple:
                module, fn = module_or_function

                if isinstance(module, nn.Module):
                    n_parameters = sum([p.shape.numel() for p in module.parameters()])
                    n_parameters_postfix = f' n_parameters: {n_parameters}'
                else:
                    n_parameters_postfix = ''

                if type(x) is tuple:
                    print(f'index: {index}, shape: {[y.shape for y in x]}' + n_parameters_postfix)
                    x = fn(module, *x)
                else:
                    print(f'index: {index}, shape: {x.shape}' + n_parameters_postfix)
                    x = fn(module, x)
            else:
                if isinstance(module_or_function, nn.Module):
                    n_parameters = sum([p.shape.numel() for p in module_or_function.parameters()])
                    n_parameters_postfix = f' n_parameters: {n_parameters}'
                else:
                    n_parameters_postfix = ''

                if type(x) is tuple:
                    print(f'index: {index}, shape: {[y.shape for y in x]}' + n_parameters_postfix)
                    x = module_or_function(*x)
                else:
                    print(f'index: {index}, shape: {x.shape}' + n_parameters_postfix)
                    x = module_or_function(x)
        return x
