import argparse
import yaml

import sklearn.datasets
import sklearn.model_selection


def prepare_dataset(config_path: str) -> None:
    """Load dataset and split into train/test.
    Args:
        config_path {str}: path of config file
    """

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    dataset = sklearn.datasets.fetch_california_housing(as_frame=True).frame

    trainset, testset = sklearn.model_selection.train_test_split(
        dataset,
        test_size=config['prepare']['test_size'],
        random_state=config['base']['random_state']
    )

    trainset.to_csv(
        path_or_buf=config['prepare']['trainset_path']
    )

    testset.to_csv(
        path_or_buf=config['prepare']['testset_path']
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    prepare_dataset(config_path=args.config)
