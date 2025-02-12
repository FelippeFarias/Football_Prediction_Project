import pickle
from datetime import datetime
from flask import current_app
import pandas as pd

def load_predictions():
    """Carrega as predições do arquivo pickle."""
    with open(current_app.config['PREDICTIONS_FILE'], 'rb') as myFile:
        pl_pred = pickle.load(myFile)
        # Garantir que a coluna Game Date seja datetime sem timezone
        pl_pred['Game Date'] = pd.to_datetime(pl_pred['Game Date']).dt.tz_localize(None)
        return pl_pred

def clean_past_predictions(pl_pred):
    """Remove predições passadas do DataFrame."""
    current_date = pd.Timestamp.now().normalize()  # Data atual sem timezone e sem hora
    # Filtrar jogos futuros
    pl_pred = pl_pred[pl_pred['Game Date'].dt.normalize() >= current_date].copy()
    return pl_pred.reset_index(drop=True) 