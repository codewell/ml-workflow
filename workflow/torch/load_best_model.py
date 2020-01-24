import os
import torch


def load_best_model(checkpoint_dir, model, optimizer=None, epoch=None):
    models = os.listdir(checkpoint_dir)
    if epoch is None:
        epoch = max([
            int(name.split('.')[0].split('_')[-1])
            for name in models
        ])
    model.load_state_dict(torch.load(
        f'{checkpoint_dir}/model_weights_{epoch}.pth',
        map_location=next(model.parameters()).device
    ))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(
            f'{checkpoint_dir}/model_optimizer_{epoch}.pth',
        ))
    return epoch
