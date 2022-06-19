import abc
import numpy as np
from utils import *
import torch

class Recommender(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def recommend(self, user_data, top_n):
        pass

    @abc.abstractmethod
    def get_sim_users(self, user_data, top_n):
        pass

class CFRecommender(Recommender):
    def __init__(self, player_data, rawid2hero):
        self.rawid2hero = rawid2hero
        self.id2rawid = list(rawid2hero.keys())
        self.rawid2id = {}
        for idx, rawid in enumerate(rawid2hero.keys()):
            self.rawid2id[rawid] = idx
        
        self.player_info = []
        rows = []
        for info, records in player_data:
            self.player_info.append(info)
            rows.append(vectorize(self.rawid2id, records))
        self.m = np.array(rows)
    
    def __cos_sim(self, v1, v2):
        return np.sum(v1 * v2) / np.sqrt(np.sum(v1 * v1) * np.sum(v2 * v2))

    def recommend(self, user_data, top_n):
        freq = data2freq(self.rawid2id, user_data)
        score = freq2score(freq)
        weight = freq / np.sum(freq)

        most_idx = np.where(score == 1)[0]
        least_idx = np.where(score == 0)[0]
        selected_idx = sorted(least_idx, 
                       key=lambda i: np.sum([self.__cos_sim(self.m[:, i], self.m[:, j]) * weight[j] for j in most_idx]) / np.sum(weight),
                       reverse=True)[:top_n]
        return list(map(lambda x: self.rawid2hero[self.id2rawid[x]], selected_idx))               
    
    def get_sim_users(self, user_data, top_n):
        u = vectorize(self.rawid2id, user_data)
        selected_idx = sorted([i for i in range(self.m.shape[0])], 
                              key=lambda x: self.__cos_sim(u, self.m[x]),
                              reverse=True)[:top_n]
        return list(map(lambda x: self.player_info[x], selected_idx))

class ACFRecommender(CFRecommender):
    def __init__(self, player_data, rawid2hero, acf_model):
        super().__init__(player_data, rawid2hero)

        self.model = acf_model
        self.device = next(self.model.parameters()).device
        self.model.eval()
        self.user_vec = self.model.forward_user()

    def recommend(self, user_data, top_n):
        score = vectorize(self.rawid2id, user_data)
        least_idx = np.where(score == 0)[0]
        
        u = torch.Tensor(score).unsqueeze(0).repeat(len(least_idx), 1).to(self.device)
        i = torch.stack([torch.LongTensor([idx]) for idx in least_idx]).to(self.device)
        pred = self.model(u, i).cpu().detach().numpy()

        selected_pairs = sorted(zip(least_idx, pred), 
                              key=lambda x: x[1], 
                              reverse=True)[:top_n]
        return list(map(lambda x: self.rawid2hero[self.id2rawid[x[0]]], selected_pairs))   