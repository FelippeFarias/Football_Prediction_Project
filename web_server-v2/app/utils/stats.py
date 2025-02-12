import pickle
from flask import current_app

def load_additional_stats():
    """Carrega as estatísticas adicionais do arquivo pickle."""
    with open(current_app.config['ADDITIONAL_STATS_FILE'], 'rb') as myFile:
        return pickle.load(myFile)

def calculate_team_stats(team_data, num_games=5):
    """Calcula estatísticas médias para um time nos últimos N jogos."""
    recent_games = team_data.tail(num_games)
    return {
        'goals_scored': recent_games['goals_scored'].mean(),
        'goals_conceded': recent_games['goals_conceded'].mean(),
        'possession': recent_games['possession'].mean(),
        'shots_on_target': recent_games['shots_on_target'].mean(),
        'shots': recent_games['shots'].mean()
    } 