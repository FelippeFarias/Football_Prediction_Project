import os

class Config:
    # Configurações básicas
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'chave-secreta-padrao'
    FLASK_APP = os.environ.get('FLASK_APP') or 'app.py'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    
    # Caminhos dos arquivos
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
    PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, 'predictions', 'results', 'predictions.pkl')
    ADDITIONAL_STATS_FILE = os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', 'prem_all_stats_dict.txt')
    
    # Configurações da API
    API_KEY_FILE = os.path.join(PROJECT_ROOT, 'api_key.txt')
    API_BASE_URL = 'https://v3.football.api-sports.io'
    API_HEADERS = {
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    # Configurações de exibição
    MAX_DISPLAY_GAMES = 40
    MAX_ADDITIONAL_DISPLAY_GAMES = 5
    
    # Debug
    DEBUG = True 