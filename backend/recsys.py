import abc
import numpy as np

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
            rows.append(self.__vectorize(records))
        self.m = np.array(rows)

    def __to_freq_vector(self, user_data):
        x = np.zeros(len(self.id2rawid))
        for hero_record in user_data:
            rawid = int(hero_record['hero_id'])
            x[self.rawid2id[rawid]] = hero_record['games']
        return x
    
    def __to_score_vector(self, v):
        x = v.copy()
        ratio = [0.4, 0.7, 0.9]
        pos = list(map(lambda z: int(x.shape[0] * z), ratio))
        sorted_idx = np.argsort(x)
        for j in sorted_idx[:pos[0]]:
            x[j] = 0
        for j in sorted_idx[pos[0]:pos[1]]:
            x[j] = 0.33
        for j in sorted_idx[pos[1]:pos[2]]:
            x[j] = 0.67
        for j in sorted_idx[pos[2]:]:
            x[j] = 1
        return x

    def __vectorize(self, user_data):
        return self.__to_score_vector(self.__to_freq_vector(user_data))
    
    def __cos_sim(self, v1, v2):
        return np.sum(v1 * v2) / np.sqrt(np.sum(v1 * v1) * np.sum(v2 * v2))

    def recommend(self, user_data, top_n):
        freq = self.__to_freq_vector(user_data)
        score = self.__to_score_vector(freq)
        weight = freq / np.sum(freq)

        most_idx = np.where(score == 1)[0]
        least_idx = np.where(score == 0)[0]
        selected_idx = sorted(least_idx, 
                       key=lambda i: np.sum([self.__cos_sim(self.m[:, i], self.m[:, j]) * weight[j] for j in most_idx]) / np.sum(weight),
                       reverse=True)[:top_n]
        return list(map(lambda x: self.rawid2hero[self.id2rawid[x]], selected_idx))               
    
    def get_sim_users(self, user_data, top_n):
        u = self.__vectorize(user_data)
        selected_idx = sorted([i for i in range(self.m.shape[0])], 
                              key=lambda x: self.__cos_sim(u, self.m[x]),
                              reverse=True)[:top_n]
        return list(map(lambda x: self.player_info[x], selected_idx))