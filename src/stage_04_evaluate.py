from src.utils.all_utils import read_yaml, create_directory, save_reports
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2


def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    test_data_filename = config["artifacts"]["test"]

    test_data_path = os.path.join(
        artifacts_dir, split_data_dir, test_data_filename)

    df = pd.read_csv(test_data_path)
    test_x = df.drop("quality", axis=1)
    test_y = df['quality']

    model_dir_name = config["artifacts"]["model_dir"]
    model_filename = config["artifacts"]["model_filename"]
    model_path = os.path.join(artifacts_dir, model_dir_name, model_filename)

    lr_model = joblib.load(model_path)

    predicted_values = lr_model.predict(test_x)

    rmse, mae, r2 = evaluate_metrics(test_y, predicted_values)
    print(rmse, " ", mae, " ", r2)

    reports_dir_name = config["artifacts"]["reports_dir"]
    reports_dir = os.path.join(artifacts_dir, reports_dir_name)
    create_directory([reports_dir])

    scores_filename = config["artifacts"]["scores"]
    scores_path = os.path.join(reports_dir, scores_filename)
    scores = {"rmse": rmse, "mae": mae, "r2": r2}

    save_reports(report=scores, report_path=scores_path, indentation=4)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    evaluate(config_path=parsed_args.config,
             params_path=parsed_args.params)
