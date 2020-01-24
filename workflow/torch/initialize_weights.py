import torch


def initialize_weights(model, init='xavier', keywords=['']):
    for name, param in model.named_parameters():
        if (
            'weight' in name
            and any(k in name for k in keywords)
            and len(param.shape) > 1
        ):
            if init == 'xavier':
                torch.nn.init.xavier_normal_(param)
            elif init == 'kaiming':
                torch.nn.init.kaiming_normal_(param)
