import numpy as np

def data2freq(rawid2id, user_data):
    x = np.zeros(len(rawid2id))
    for hero_record in user_data:
        rawid = int(hero_record['hero_id'])
        x[rawid2id[rawid]] = hero_record['games']
    return x

def freq2score(v):
    x = v.copy()
    ratio = [0.5, 0.7, 0.9]
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

def vectorize(rawid2id, user_data):
    return freq2score(data2freq(rawid2id, user_data))