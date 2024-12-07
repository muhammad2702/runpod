import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration
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


# Timeframes you want to collect data for
TIMEFRAMES = [

    {'multiplier': 1, 'timespan': 'second'},
]

# Date range for data collection
START_DATE = '2024-05-01'
END_DATE = '2024-12-05'

# Directory to save the collected data
DATA_DIR = 'crypto_data'
os.makedirs(DATA_DIR, exist_ok=True)

def daterange(start_date, end_date, delta):
    current = start_date
    while current <= end_date:
        yield current
        current += delta

def fetch_data(ticker, multiplier, timespan, from_date, to_date):
    url = BASE_URL.format(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date.strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d')
    )
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': '5000',  # Maximum allowed by Polygon.io
        'apiKey': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        print(f"Error fetching data for {ticker} - {timespan}: {response.status_code} - {response.text}")
        return []

def collect():
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.strptime(END_DATE, '%Y-%m-%d')

    for ticker in TICKERS:
        ticker_dir = os.path.join(DATA_DIR, ticker.replace(":", "_"))
        os.makedirs(ticker_dir, exist_ok=True)

        for timeframe in TIMEFRAMES:
            multiplier = timeframe['multiplier']
            timespan = timeframe['timespan']
            filename = f"{ticker.replace(':', '_')}_{multiplier}{timespan}.csv"
            filepath = os.path.join(ticker_dir, filename)

            print(f"Fetching data for {ticker} - {multiplier}{timespan}")

            # To handle large date ranges, you might need to split the requests
            # For simplicity, we'll attempt to fetch all data at once
            data = fetch_data(ticker, multiplier, timespan, start, end)

            if data:
                df = pd.DataFrame(data)
                # Convert timestamp to datetime
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                # Save to CSV
                df.to_csv(filepath, index=False)
                print(f"Saved {len(df)} records to {filepath}")
            else:
                print(f"No data fetched for {ticker} - {multiplier}{timespan}")

            # Respect API rate limits
            time.sleep(1)  # Adjust sleep time based on your rate limits

if __name__ == "__main__":
    collect()
