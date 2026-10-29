from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelTrainer(ABC):
    """Abstraction commune à tous les entraîneurs de modèles."""

    @abstractmethod
    def train(self, features, target):
        """Entraîne et retourne un modèle."""


class RandomForestTrainer(ModelTrainer):
    def train(self, features, target):
        model = RandomForestClassifier(random_state=42)
        model.fit(features, target)
        return model


class LogisticRegressionTrainer(ModelTrainer):
    def train(self, features, target):
        model = LogisticRegression(max_iter=1000)
        model.fit(features, target)
        return model
