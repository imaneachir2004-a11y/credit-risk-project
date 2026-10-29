# Credit Risk

Prédiction du risque de défaut de crédit (Random Forest / Régression logistique).

## Installation

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e . ruff black pytest

## Lancer le projet

python main.py

## Qualité du code

ruff check .
black --check .
## Auteure

Achir Imane, groupe 3, EMSI.

## Structure

- `src/credit_risk/` : modules (data, preprocessing, training, evaluation, prediction, trainers, experiment)
- `data/raw/` : jeu de données
- `notebooks/` : exploration
- `tests/` : tests
- `docs/` : compte rendu du TP