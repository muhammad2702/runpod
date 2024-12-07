from celery import Celery
import subprocess

app = Celery('crypto_tasks')

@app.task
def trigger_data_collection():
    subprocess.run(["python", "data_collector.py"])

@app.task
def trigger_preprocessing():
    subprocess.run(["python", "preprocessor.py"])

@app.task
def trigger_training():
    subprocess.run(["python", "model_trainer.py"])

@app.task
def trigger_prediction():
    subprocess.run(["python", "predictor.py"])

