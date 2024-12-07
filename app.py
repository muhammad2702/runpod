from flask import Flask, request, jsonify, render_template
import subprocess
from celery_tasks import trigger_data_collection, trigger_preprocessing, trigger_training, trigger_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect_data', methods=['POST'])
def collect_data():
    trigger_data_collection.delay()
    return jsonify({"message": "Data collection started."})

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    trigger_preprocessing.delay()
    return jsonify({"message": "Data preprocessing started."})

@app.route('/train_model', methods=['POST'])
def train_model():
    trigger_training.delay()
    return jsonify({"message": "Model training started."})

@app.route('/predict', methods=['POST'])
def predict():
    trigger_prediction.delay()
    return jsonify({"message": "Prediction job started."})

if __name__ == '__main__':
    app.run(debug=True)

