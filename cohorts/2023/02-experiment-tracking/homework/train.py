import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#import mlflow and mlflow.sklearn
import mlflow
import mlflow.sklearn

#set tracking uri
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Q3_Autolog_Experiment_RF")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="/opt/mlops/mlops-zoomcamp/cohorts/2023/data/output/",
    help="Location where the processed NYC taxi trip data was saved"
)


def run_train(data_path: str):
    #enable autologging
    mlflow.sklearn.autolog()   
    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)


if __name__ == '__main__':
    run_train()
