from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

bundle = joblib.load('ipl_model.pkl')
model  = bundle['model']
enc    = bundle['encoders']

def safe_encode(encoder, value):
    if value not in list(encoder.classes_):
        return 0
    return int(encoder.transform([value])[0])

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    features = [[
        safe_encode(enc['batting_team'],  d['batting_team']),
        safe_encode(enc['bowling_team'],  d['bowling_team']),
        safe_encode(enc['venue'],         d['venue']),
        safe_encode(enc['toss_winner'],   d['toss_winner']),
        safe_encode(enc['toss_decision'], d['toss_decision']),
        safe_encode(enc['stage'],         d['stage']),
        1 if d['toss_winner'] == d['batting_team'] and d['toss_decision'] == 'bat' else 0
    ]]
    proba  = model.predict_proba(features)[0]
    winner = d['bat
