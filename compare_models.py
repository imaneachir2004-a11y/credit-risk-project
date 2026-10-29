from sklearn.model_selection import train_test_split

from credit_risk.data import load_data
from credit_risk.experiment import run_experiment
from credit_risk.preprocessing import clean_data, split_features_target
from credit_risk.trainers import LogisticRegressionTrainer, RandomForestTrainer


def main():
    data = clean_data(load_data("data/raw/credit.csv"))
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    for trainer in (RandomForestTrainer(), LogisticRegressionTrainer()):
        _, score = run_experiment(trainer, x_train, y_train, x_test, y_test)
        print(f"{type(trainer).__name__} : {score:.3f}")


if __name__ == "__main__":
    main()
