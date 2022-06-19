from flask import Flask, Response, request
from flask_cors import CORS
import requests
import json
import pickle as pkl
from recsys import CFRecommender, ACFRecommender
from model import ACF4HeroRec
import numpy as np
import torch

app = Flask(__name__)
CORS(app, supports_credentials=True)

with open('../rawid2hero.pkl', 'rb') as f:
    rawid2hero = pkl.load(f)
with open('../player_composed.pkl', 'rb') as f:
    player_list = pkl.load(f)
kg_embed = np.load('../kg_embed.npy')
model = ACF4HeroRec(25, kg_embed)
model.load_state_dict(torch.load('ckpts/30.pth'))
model = model.cuda()
rec = ACFRecommender(player_list, rawid2hero, model)
#rec = CFRecommender(player_list, rawid2hero)

@app.route('/get_result', methods=['GET', 'POST'])
def get_result(): 
    uid = request.args.get('uid')
    top_p, top_r, top_s = int(request.args.get('top_p')), int(request.args.get('top_r')), int(request.args.get('top_s'))
    data = eval(requests.get('https://api.opendota.com/api/players/{}/heroes'.format(uid)).text)
    if not len(data):
        res = {
            "preference": [],
            "recommended_heroes": [],
            "similar_players": [],
        }
        return Response(json.dumps(res), mimetype='application/json')
    total_games = np.sum([int(d['games']) for d in data])
    preference = sorted(data, key=lambda x: x['games'], reverse=True)[:int(top_p)]
    preference = list(filter(lambda x: x['games'], preference))
    for x in preference:
        x['localized_name'] = rawid2hero[int(x['hero_id'])]['localized_name']
        x['win_rate'] = '{:.2%}'.format(float(x['win']) / float(x['games']))
        x['preference_factor'] = '{:.2f}'.format(float(x['games']) / total_games)
    rec_res = rec.recommend(data, top_r)
    player_res = rec.get_sim_users(data, top_s)

    res = {
        "preference": preference,
        "recommended_heroes": rec_res,
        "similar_players": player_res,
    }
    return Response(json.dumps(res), mimetype='application/json')



if __name__ == '__main__':
    app.run(debug=True)