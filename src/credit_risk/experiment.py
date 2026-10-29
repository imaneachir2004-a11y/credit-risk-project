from credit_risk.evaluation import evaluate_model
from credit_risk.trainers import ModelTrainer


def run_experiment(trainer: ModelTrainer, x_train, y_train, x_test, y_test):
    """Entraîne et évalue n'importe quel modèle fourni par un trainer."""
    model = trainer.train(x_train, y_train)
    return model, evaluate_model(model, x_test, y_test)
