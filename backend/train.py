import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim 
import pickle as pkl
import numpy as np
from model import ACF4HeroRec
from dataset import HeroRecDataset

with open('../rawid2hero.pkl', 'rb') as f:
    r = pkl.load(f)
with open('../player_composed.pkl', 'rb') as f:
    pd = pkl.load(f)

dataset = HeroRecDataset(pd, r)
train_size = int(len(dataset) * 0.6)
val_size = len(dataset) - train_size
trainset, valset = random_split(dataset, [train_size, val_size])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

hero_embed = np.load('../kg_embed.npy')
model = ACF4HeroRec(latent_dim=25, hero_embed=hero_embed).to(device)

train_loader = DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True)
#train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=1)

train_epoch = 50
val_interval = 1

def train():
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(1, train_epoch + 1):
        model.train()
        losses = []
        for u, i, y in train_loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            pred = model(u, i)
            loss = criterion(pred, y)
            losses.append(loss.cpu().detach().numpy() ** 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % val_interval == 0:
            print('Epoch {}: mean_train_rmse={} mean_eval_rmse={} eval_acc={}'.format(epoch, 
                                                                                      np.mean(losses), 
                                                                                      *validate()))
            #print('Epoch {}: mean_train_rmse={}'.format(epoch, np.mean(losses)))
            torch.save(model.state_dict(), 'ckpts/t{}.pth'.format(epoch))
    return model

def validate():
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    gt, p = [], []
    for u, i, y in val_loader:
        u, i, y = u.to(device), i.to(device), y.to(device)
        pred = model(u, i)
        loss = criterion(pred, y)
        losses.append(loss.cpu().detach().numpy() ** 0.5)
        p.append(pred.cpu().detach().numpy()[0])
        gt.append(y.cpu()[0])
    
    labels = np.array([0, 0.33, 0.67, 1])
    correct = []
    eq = lambda x, y: np.abs(x - y) < 1e-5
    for i in range(len(gt)):
        correct.append(eq(gt[i], labels[np.argmin(np.abs(labels - p[i]))]))

    return np.mean(losses), np.mean(correct)

train()