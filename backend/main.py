from flask import Flask, Response, request
from flask_cors import CORS
import requests
import json
import pickle as pkl
from recsys import CFRecommender

app = Flask(__name__)
CORS(app, supports_credentials=True)

with open('../rawid2hero.pkl', 'rb') as f:
    rawid2hero = pkl.load(f)
with open('../player_composed.pkl', 'rb') as f:
    player_list = pkl.load(f)

rec = CFRecommender(player_list, rawid2hero)

@app.route('/most_played', methods=['GET', 'POST'])
def most_played(): 
    uid, top_n = request.args.get('uid'), request.args.get('top_n')
    data = eval(requests.get('https://api.opendota.com/api/players/{}/heroes'.format(uid)).text)
    preference = sorted(data, key=lambda x: x['games'], reverse=True)[:int(top_n)]
    preference = list(filter(lambda x: x['games'], preference))
    for x in preference:
        x['localized_name'] = rawid2hero[int(x['hero_id'])]['localized_name']
    
    rec_res = rec.recommend(data, 3)
    player_res = rec.get_sim_users(data, 3)

    res = {
        "preference": preference,
        "recommended_heroes": rec_res,
        "similar_players": player_res,
    }
    return Response(json.dumps(res), mimetype='application/json')



if __name__ == '__main__':
    app.run(debug=True)