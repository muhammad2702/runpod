import os
import time
import requests
import pandas as pd
import torch
from datetime import datetime, timedelta
from preprocessor import CryptoDataPreprocessor
from model_trainer import ShortTermTransformerModel

API_KEY = 'de_kgSuhw6v4KnRK0wprJCoBAIhqSd5R'  # Replace with your actual API key
BASE_URL = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'

# List of cryptocurrencies (tickers) you want to collect data for
TICKERS = [
    'X:AAVEUSD',
    'X:AVAXUSD',
    'X:BATUSD',
    'X:LINKUSD',
    'X:UNIUSD',
    'X:SUSHIUSD',
    'X:PNGUSD',
    'X:JOEUSD',
    'X:XAVAUSD',
    'X:ATOMUSD',
    'X:ALGOUSD',
    'X:ARBUSD',
    'X:1INCHUSD',
    'X:DAIUSD',
    # Add more tickers as needed
]

def fetch_latest_data():
    """
    Fetch the latest real-time data for all tickers.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=5)  # Fetch data for the last 5 minutes

    latest_data = []
    for ticker in TICKERS:
        url = BASE_URL.format(
            ticker=ticker,
            multiplier=1,
            timespan='minute',
            from_date=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            to_date=end_date.strftime('%Y-%m-%dT%H:%M:%S')
        )
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': '5000',
            'apiKey': API_KEY
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json().get('results', [])
            for record in data:
                record['ticker'] = ticker
            latest_data.extend(data)
        else:
            print(f"Error fetching data for {ticker}: {response.status_code} - {response.text}")

    # Convert to DataFrame
    if latest_data:
        df = pd.DataFrame(latest_data)
        df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp
        return df
    else:
        raise ValueError("No data fetched for tickers.")

def preprocess_and_predict():
    """
    Fetch latest data, preprocess it, and use the model to make predictions.
    """
    # Step 1: Fetch latest data
    print("Fetching latest data...")
    latest_data = fetch_latest_data()
    print(f"Fetched {len(latest_data)} records.")

    # Step 2: Preprocess the data
    print("Preprocessing the data...")
    preprocessor = CryptoDataPreprocessor(
        raw_data_dir=None,  # Not needed for real-time data
        preprocessed_data_dir=None  # Not needed for real-time data
    )
    preprocessed_data, _ = preprocessor.preprocess_file(latest_data)

    # Step 3: Load the trained model
    print("Loading the model...")
    model = ShortTermTransformerModel(
        num_features=len(preprocessed_data.columns) - 1,  # Exclude target columns
        num_cryptos=len(TICKERS),
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=128
    )
    model.load_state_dict(torch.load('best_short_term_transformer_model.pth'))
    model.eval()

    # Step 4: Prepare data for prediction
    inputs = torch.tensor(preprocessed_data.iloc[:, :-1].values, dtype=torch.float32)  # Exclude target
    crypto_ids = torch.tensor([TICKERS.index(ticker) for ticker in latest_data['ticker']], dtype=torch.long)

    # Step 5: Make predictions
    print("Making predictions...")
    percent_probs, leg_probs = model(inputs, crypto_ids)
    predictions = {
        "percent_change": percent_probs.argmax(dim=1).tolist(),
        "leg_direction": leg_probs.argmax(dim=1).tolist()
    }

    # Output results
    print("Predictions:")
    print(predictions)

    # Save predictions for visualization
    preprocessed_data['percent_change_prediction'] = predictions["percent_change"]
    preprocessed_data['leg_direction_prediction'] = predictions["leg_direction"]
    preprocessed_data.to_csv('predictions/latest_predictions.csv', index=False)
    print("Predictions saved to 'predictions/latest_predictions.csv'.")

if __name__ == "__main__":
    preprocess_and_predict()

