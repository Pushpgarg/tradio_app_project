import pandas as pd

data = pd.read_csv('data/nifty50_ticks.csv')

# symbol, volume, open_interest, exchange contains same value accross all rows
data.drop(columns=['id', 'symbol','volume','open_interest', 'exchange'] , inplace=True)

# sorting by timestamp as in for the label generation we will need data in chronological order
data.sort_values(by='timestamp', inplace=True, ignore_index=True)

# splitting timestamp into year, month, day, hour, minute columns to pass data to the model
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month
data['day'] = data['timestamp'].dt.day
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute

data.drop(columns=['timestamp'], inplace=True)

# Label generation
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

data.to_csv('./data/processed_data.csv', index=False)