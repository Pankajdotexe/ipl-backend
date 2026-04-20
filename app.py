from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

bundle = joblib.load('ipl_model.pkl')
model = bundle['model']
enc = bundle['encoders']

def safe_encode(encoder, value):
    if value not in list(encoder.classes_):
        return 0
    return int(encoder.transform([value])[0])

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    bat = d['batting_team']
    bowl = d['bowling_team']
    features = [[
        safe_encode(enc['batting_team'], bat),
        safe_encode(enc['bowling_team'], bowl),
        safe_encode(enc['venue'], d['venue']),
        safe_encode(enc['toss_winner'], d['toss_winner']),
        safe_encode(enc['toss_decision'], d['toss_decision']),
        safe_encode(enc['stage'], d['stage']),
        1 if d['toss_winner'] == bat and d['toss_decision'] == 'bat' else 0
    ]]
    proba = model.predict_proba(features)[0]
    winner = bat if proba[1] > 0.5 else bowl
    return jsonify({
        'predicted_winner': winner,
        'batting_team_win_prob': round(float(proba[1]) * 100, 1),
        'bowling_team_win_prob': round(float(proba[0]) * 100, 1),
        'confidence': 'High' if max(proba) > 0.7 else 'Medium' if max(proba) > 0.55 else 'Low'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run()
