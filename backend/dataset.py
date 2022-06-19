from torch.utils.data import Dataset
import torch
from utils import *
import pickle as pkl

class HeroRecDataset(Dataset):
    def __init__(self, player_data, rawid2hero):
        rawid2id = {}
        for idx, rawid in enumerate(rawid2hero.keys()):
            rawid2id[rawid] = idx

        n = len(rawid2id)
        self.data = []

        for _, records in player_data:
            v = vectorize(rawid2id, records)
            for i in np.where(v > 0)[0]:
                user_score = torch.FloatTensor(v)
                user_score[i] = 0
                item_idx = torch.LongTensor([i])
                y = v[i].astype(np.float32)
                self.data.append({
                    'user_score': user_score,
                    'item_idx': item_idx,
                    'y': y
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return list(self.data[idx].values())

if __name__ == '__main__':
    with open('../rawid2hero.pkl', 'rb') as f:
        r = pkl.load(f)
    with open('../player_composed.pkl', 'rb') as f:
        pd = pkl.load(f)
    ds = HeroRecDataset(pd, r)
    print(ds[10])