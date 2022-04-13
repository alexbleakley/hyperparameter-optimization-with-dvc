import argparse
import json
import yaml

import joblib
import pandas as pd
import sklearn.metrics


def evaluate_model(config_path: str) -> None:
    """Evaluate model.
    Args:
        config_path {str}: path of config file
    """

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    model = joblib.load(
        filename=config['train']['model_path']
    )

    testset = pd.read_csv(
        filepath_or_buffer=config['prepare']['testset_path']
    )

    X = testset.drop(columns=['Unnamed: 0', 'MedHouseVal'], inplace=False)
    y = testset['MedHouseVal']

    prediction = model.predict(X)

    r_square = sklearn.metrics.r2_score(y_true=y, y_pred=prediction)
    mse = sklearn.metrics.mean_squared_error(y_true=y, y_pred=prediction)
    mae = sklearn.metrics.mean_absolute_error(y_true=y, y_pred=prediction)

    report = {
        'r_square': r_square,
        'mse': mse,
        'mae': mae
    }

    with open(config['evaluate']['metrics_path'], 'w') as metrics_file:
        json.dump(
            obj=report,
            fp=metrics_file
        )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
