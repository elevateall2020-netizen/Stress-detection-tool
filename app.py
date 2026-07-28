"""
MINDFIELD Flask API
Serves ML predictions from model.pkl (XGBClassifier, 18-feature xhat vector).
Returns stress level (0=Low, 1=Medium, 2=High).
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return send_from_directory('.', 'mindfield.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        xhat = data.get('xhat')
        if xhat is None or len(xhat) != 18:
            return jsonify({'error': 'xhat must be an array of exactly 18 numbers'}), 400
        xhat_arr = np.array(xhat, dtype=float).reshape(1, -1)
        level = int(model.predict(xhat_arr)[0])
        proba = model.predict_proba(xhat_arr)[0].tolist()
        return jsonify({'level': level, 'probabilities': proba})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'XGBClassifier', 'features': 18})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
