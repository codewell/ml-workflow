

def set_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = learning_rate
        param_group['lr'] = learning_rate
