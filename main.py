from sklearn.model_selection import train_test_split

from credit_risk.data import load_data
from credit_risk.evaluation import evaluate_model
from credit_risk.preprocessing import clean_data, split_features_target
from credit_risk.training import train_model


def main():
    data = clean_data(load_data("data/raw/credit.csv"))
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = train_model(x_train, y_train)
    print(f"Accuracy (test) : {evaluate_model(model, x_test, y_test):.3f}")


if __name__ == "__main__":
    main()
