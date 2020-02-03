import torch


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, fn, dataframe):
        super().__init__()
        self.fn = fn
        self.dataframe = dataframe
        self.size = len(dataframe)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.fn(self.dataframe.loc[index])
