import argparse
import yaml

import joblib
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing


def train_model(config_path: str) -> None:
    """Train model.
    Args:
        config_path {str}: path of config file
    """

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    trainset = pd.read_csv(
        filepath_or_buffer=config['prepare']['trainset_path']
    )

    normalizer = sklearn.preprocessing.StandardScaler()
    estimator = sklearn.linear_model.ElasticNet(
        alpha=config['train']['alpha'],
        l1_ratio=config['train']['l1_ratio'],
        random_state=config['base']['random_state']
    )
    pipeline = sklearn.pipeline.Pipeline([
        ('normalizer', normalizer),
        ('estimator', estimator)
    ])

    X = trainset.drop(columns=['Unnamed: 0', 'MedHouseVal'], inplace=False)
    y = trainset['MedHouseVal']

    model = pipeline.fit(X, y)

    joblib.dump(
        model,
        filename=config['train']['model_path']
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
