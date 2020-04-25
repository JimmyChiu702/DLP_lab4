import torch
from torch.utils.data import Dataset
import json


def getData(mode):
    if mode == 'train':
        filename = 'train.json'
    else:
        filename = 'test.json'
    with open(filename, 'r') as f:
        return json.load(f)
        

class Seq2seqDataset(Dataset):
    def __init__(self, mode, transform):
        data = getData(mode)
        self.mode = mode
        self.pairs = []
        for d in data:
            for i in d['input']:
                self.pair.append((i, d['output']))
        self.transform = transform

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolost()

        return self.transform(self.pair[idx])