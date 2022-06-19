import torch
from torch import nn 
import numpy as np
import torch.nn.functional as F

class ACF4HeroRec(nn.Module):
    def __init__(self, latent_dim=25, hero_embed=None):
        super().__init__()
        
        assert isinstance(hero_embed, np.ndarray)

        n, d = hero_embed.shape
        self.hero_embed = nn.Parameter(torch.Tensor(hero_embed), 
                                       requires_grad=False)
        self.interacted_item_embed = nn.Parameter(torch.Tensor(n, latent_dim))
        nn.init.xavier_uniform_(self.interacted_item_embed)
        self.item_embed = nn.Embedding(n, latent_dim)
        nn.init.xavier_uniform_(self.item_embed.weight)

        self.c_attn_layers = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Softmax(dim=1)
        )

        self.w1xb1 = nn.Linear(d, latent_dim)
        self.w1p = nn.Linear(latent_dim, latent_dim, bias=False)
        self.w1v = nn.Linear(latent_dim, latent_dim, bias=False)
        self.i_attn_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

        self.cls = nn.Linear(1, 1)

    def forward_user(self):
        x = self.hero_embed
        c_attn = self.c_attn_layers(x)
        x = c_attn * x
        
        v = self.item_embed.weight
        p = self.interacted_item_embed
        i_attn = self.w1xb1(x) + self.w1p(p) + self.w1v(v)
        i_attn = self.i_attn_layers(i_attn)
        i_attn = F.softmax(i_attn, dim=0)
        user_vec = i_attn * p

        return user_vec

    def forward_score(self, user_vec, u_scores, i_indices):
        user_vec_batch = user_vec.unsqueeze(0) * u_scores.unsqueeze(2)
        user_vec_batch = torch.sum(user_vec_batch, dim=1)
        item_vec_batch = self.item_embed(i_indices).squeeze(1)
        
        preds = torch.sigmoid(self.cls(torch.sum(user_vec_batch * item_vec_batch, dim=1, keepdim=True))) 
        return preds.squeeze(1)
    
    def forward(self, u_scores, i_indices):        
        user_vec = self.forward_user()
        preds = self.forward_score(user_vec, u_scores, i_indices)
        return preds
