import os
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

class CryptoDataPreprocessor:
    def __init__(self, raw_data_dir='crypto_data', preprocessed_data_dir='preprocessed_data', columns_to_add=None):
        """
        Initializes the CryptoDataPreprocessor.

        :param raw_data_dir: Directory containing raw CSV data.
        :param preprocessed_data_dir: Directory to save preprocessed data.
        :param columns_to_add: List of columns to include in the final output.
        """
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.columns_to_add = columns_to_add or ['leg_direction', 'close_price']  # Default columns
        os.makedirs(self.preprocessed_data_dir, exist_ok=True)
        self.label_encoders = {}

    def preprocess_file(self, df):
        """
        Applies preprocessing steps to the DataFrame.

        :param df: pandas DataFrame with raw data.
        :return: Preprocessed DataFrame and label encoders.
        """
        required_columns = ['c', 'h', 'l']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Compute RSI
        rsi = momentum.RSIIndicator(close=df['c'], window=14)
        df['RSI'] = rsi.rsi()

        # Compute MACD with smoothing
        macd = trend.MACD(close=df['c'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd().rolling(window=3).mean()
        df['MACD_signal'] = macd.macd_signal().rolling(window=3).mean()
        df['MACD_diff'] = macd.macd_diff().rolling(window=3).mean()

        # Compute ATR
        atr = volatility.AverageTrueRange(high=df['h'], low=df['l'], close=df['c'], window=14)
        df['ATR'] = atr.average_true_range().rolling(window=3).mean()

        # Compute Bollinger Bands Width
        bollinger = volatility.BollingerBands(close=df['c'], window=20, window_dev=2)
        df['BB_upper'] = bollinger.bollinger_hband().rolling(window=3).mean()
        df['BB_lower'] = bollinger.bollinger_lband().rolling(window=3).mean()
        df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['c']).rolling(window=3).mean()

        # Compute ADX with smoothing
        adx = trend.ADXIndicator(high=df['h'], low=df['l'], close=df['c'], window=14)
        df['ADX'] = adx.adx().rolling(window=3).mean()

        # Drop initial rows with NaN values
        df.dropna(inplace=True)

        # Classify Market Environments
        df, label_encoders = self.classify_market_environments(df)

        # Calculate Leg Data
        df = self.calculate_leg_data(df)

        # Classify Percent Change
        df = self.classify_percent_change(df)
        df['close_price'] = df['c']  # Assuming 'c' is the closing price

        return df, label_encoders

    def classify_market_environments(self, df):
        """
        Classify market environments into numerical categories for model training.

        :param df: pandas DataFrame with technical indicators.
        :return: DataFrame with new classification columns and label encoders.
        """
        df['ATR_mavg'] = df['ATR'].rolling(window=14).mean()
        df['ATR_vol'] = np.where(df['ATR'] > 1.2 * df['ATR_mavg'], 'h',
                                 np.where(df['ATR'] < 0.8 * df['ATR_mavg'], 'l', 'Medium'))

        df['BB_mavg'] = df['BB_width'].rolling(window=20).mean()
        df['BB_vol'] = np.where(df['BB_width'] > 1.2 * df['BB_mavg'], 'h',
                                np.where(df['BB_width'] < 0.8 * df['BB_mavg'], 'l', 'Medium'))

        df['daily_return'] = df['c'].pct_change()
        df['RV'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        rv_80 = df['RV'].quantile(0.8)
        rv_20 = df['RV'].quantile(0.2)
        df['RV_vol'] = np.where(df['RV'] > rv_80, 'h',
                                np.where(df['RV'] < rv_20, 'l', 'Medium'))

        df['Volatility'] = df[['ATR_vol', 'BB_vol', 'RV_vol']].mode(axis=1)[0]

        df['Trend'] = np.where(
            (df['MACD'] > df['MACD_signal']) & (df['MACD'] > 0), 'Bullish',
            np.where(
                (df['MACD'] < df['MACD_signal']) & (df['MACD'] < 0), 'Bearish', 'Neutral'
            )
        )

        df['Trend_strength'] = np.where(df['ADX'] > 25, 'Strong', 'Weak')

        df['Market_Environment'] = df.apply(
            lambda row: f"{row['Volatility']} Vol/{row['Trend']}" if row['Trend_strength'] == 'Strong' else f"{row['Volatility']} Vol/Neutral",
            axis=1
        )

        label_encoders = {}
        categorical_columns = ['ATR_vol', 'BB_vol', 'RV_vol', 'Volatility', 'Trend', 'Trend_strength', 'Market_Environment']
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

        self.label_encoders = label_encoders
        return df, label_encoders

    def calculate_leg_data(self, df):
        df['percent_delta'] = df['c'].pct_change()
        df = df.reset_index(drop=True)

        previous_leg_change = 0
        previous_leg_length = 0
        current_leg_change = 0
        current_leg_length = 0
        current_direction = None

        previous_changes = []
        previous_lengths = []
        current_changes = []
        current_lengths = []
        leg_directions = []

        for i in range(len(df)):
            if i == 0:
                current_leg_change = 0
                current_leg_length = 0
                current_direction = 0  # Neutral at start
            else:
                percent_delta = df.at[i, 'percent_delta']
                if current_leg_length == 0:
                    current_leg_change = percent_delta
                    current_leg_length = 1
                    current_direction = 1 if percent_delta > 0 else 0
                else:
                    if (current_leg_change > 0 and percent_delta > 0) or (current_leg_change < 0 and percent_delta < 0):
                        current_leg_change += percent_delta
                        current_leg_length += 1
                    else:
                        previous_leg_change = current_leg_change
                        previous_leg_length = current_leg_length
                        current_leg_change = percent_delta
                        current_leg_length = 1
                        current_direction = 1 if percent_delta > 0 else 0
            previous_changes.append(previous_leg_change)
            previous_lengths.append(previous_leg_length)
            current_changes.append(current_leg_change)
            current_lengths.append(current_leg_length)
            leg_directions.append(current_direction)

        df['previous_leg_change'] = previous_changes
        df['previous_leg_length'] = previous_lengths
        df['current_leg_change'] = current_changes
        df['current_leg_length'] = current_lengths
        df['leg_direction'] = leg_directions

        df.drop(columns=['percent_delta'], inplace=True)
        return df

    def classify_percent_change(self, df):
        df['percent_change'] = df['c'].pct_change()
        df.dropna(inplace=True)

        percentiles = df['percent_change'].quantile([0.05, 0.20, 0.40, 0.60, 0.80, 0.95]).to_dict()

        def classify(x, p):
            if x < p[0.05]:
                return 'Down a Lot'
            elif x < p[0.20]:
                return 'Down Moderate'
            elif x < p[0.40]:
                return 'Down a Little'
            elif x < p[0.60]:
                return 'No Change'
            elif x < p[0.80]:
                return 'Up a Little'
            elif x < p[0.95]:
                return 'Up Moderate'
            else:
                return 'Up a Lot'

        df['percent_change_classification'] = df['percent_change'].apply(lambda x: classify(x, percentiles))

        le = LabelEncoder()
        df['percent_change_classification'] = le.fit_transform(df['percent_change_classification'].astype(str))
        self.label_encoders['percent_change_classification'] = le

        return df

    def save_preprocessed_data(self, df, filepath):
        """
        Saves the preprocessed DataFrame to a CSV file with selected columns.

        :param df: pandas DataFrame with preprocessed data.
        :param filepath: Path where the CSV will be saved.
        """

        df[self.columns_to_add].to_csv(filepath, index=False)
        print(f"Saved preprocessed data to {filepath}")

    def preprocess_all_files(self):
    # Traverse the raw_data_dir
      for ticker in tqdm(os.listdir(self.raw_data_dir), desc='Processing Tickers'):
          ticker_raw_dir = os.path.join(self.raw_data_dir, ticker)
          ticker_preprocessed_dir = os.path.join(self.preprocessed_data_dir, ticker)
          os.makedirs(ticker_preprocessed_dir, exist_ok=True)

          for file in tqdm(os.listdir(ticker_raw_dir), desc=f'Processing {ticker}', leave=False):
              if file.endswith('.csv'):
                  raw_filepath = os.path.join(ticker_raw_dir, file)
                  preprocessed_filename = file.replace('.csv', '_preprocessed.csv')
                  preprocessed_filepath = os.path.join(ticker_preprocessed_dir, preprocessed_filename)

                  if os.path.exists(preprocessed_filepath):
                      print(f"Preprocessed file already exists: {preprocessed_filepath}. Skipping.")
                      continue

                  # Read raw CSV
                  try:
                      df_raw = pd.read_csv(raw_filepath)
                      # Ensure timestamp is datetime if needed
                      if 'timestamp' in df_raw.columns and not pd.api.types.is_datetime64_any_dtype(df_raw['timestamp']):
                          df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                  except Exception as e:
                      print(f"Error reading {raw_filepath}: {e}")
                      continue

                # Preprocess
                  try:
                      df_preprocessed, label_encoders = self.preprocess_file(df_raw)  # Unpack the tuple
                  except Exception as e:
                      print(f"Error preprocessing {raw_filepath}: {e}")
                      continue

                  # Print head of the DataFrame
                  print(f"Preprocessed data for {file}:")
                  print(df_preprocessed.head())  # Print the entire head without truncation

                  # Save preprocessed data
                  self.save_preprocessed_data(df_preprocessed, preprocessed_filepath)



def preprocess():
    """
    Main function to preprocess cryptocurrency data.
    """
    # Define directories
    raw_data_dir = 'crypto_data'
    preprocessed_data_dir = 'preprocessed_data'

    # Columns to include in the final output
    columns_to_add = [
        'leg_direction', 'close_price', 'o' ,'l',  'h', 't' ,'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'ATR', 'BB_width', 'ADX' , #, 'Volatility', 'Trend', 'Trend_strength',
        'Market_Environment', 'percent_change_classification' #'previous_leg_change', 'previous_leg_length',
        #'current_leg_change', 'current_leg_length'
    ]

    # Initialize the preprocessor
    preprocessor = CryptoDataPreprocessor(
        raw_data_dir=raw_data_dir,
        preprocessed_data_dir=preprocessed_data_dir,
        columns_to_add=columns_to_add
    )

    # Process all files
    print("Starting preprocessing...")
    preprocessor.preprocess_all_files()
    print("Preprocessing completed.")


# Run the main function
if __name__ == '__main__':
    preprocess()


