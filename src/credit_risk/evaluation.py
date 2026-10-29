def evaluate_model(model, features, target):
    """Retourne le score du modèle."""
    return model.score(features, target)
