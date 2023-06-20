import pickle
import pandas as pd
import argparse

def read_data(url):
    df = pd.read_parquet(url)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(url, dv, model):
    df = read_data(url)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print("std_dev=", y_pred.std())
    print("mean=", y_pred.mean())
    return y_pred

def load_model(model_file):
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

# add 2 argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2022)
parser.add_argument('--month', type=int, default=2)
args = parser.parse_args()

#pass these args to url
url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month:02d}.parquet'

if __name__ == "__main__":

    categorical = ['PULocationID', 'DOLocationID']
    dv, model = load_model('model.bin')
    
    predict(url, dv, model)
    



