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
        random_state=config['base']['random_state']
    )
    pipeline = sklearn.pipeline.Pipeline([
        ('normalizer', normalizer),
        ('estimator', estimator)
    ])

    kfold_cv = sklearn.model_selection.KFold(
        n_splits=config['train']['kfold_n_splits'],
        shuffle=True,
        random_state=config['base']['random_state']
    )

    param_grid = {
        'estimator__alpha': config['train']['alpha'],
        'estimator__l1_ratio': config['train']['l1_ratio']
    }

    optimizer = sklearn.model_selection.GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kfold_cv,
        scoring=config['train']['scoring_function']
    )

    X = trainset.drop(columns=['Unnamed: 0', 'MedHouseVal'], inplace=False)
    y = trainset['MedHouseVal']

    model = optimizer.fit(X, y)

    joblib.dump(
        model,
        filename=config['train']['model_path']
    )

    hpo_metrics = pd.DataFrame(optimizer.cv_results_)

    hpo_metrics.to_json(
        path_or_buf=config['train']['hpo_metrics_path']
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
