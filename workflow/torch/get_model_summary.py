import pandas as pd


def get_model_summary(model):
    params_summary = pd.DataFrame(
        [[n, p.numel()] for n, p in model.named_parameters()],
        columns=['name', '# params']
    )
    num_params = params_summary['# params'].sum()
    params_summary['# params'] = list(map('{:,}'.format,
        params_summary['# params']))
    return params_summary, num_params
