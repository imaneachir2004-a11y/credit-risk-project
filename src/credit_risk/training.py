from sklearn.ensemble import RandomForestClassifier


def train_model(features, target):
    """Entraîne un modèle Random Forest."""
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)
    return model
