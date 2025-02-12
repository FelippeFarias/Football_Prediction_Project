import requests
from flask import current_app
import json
import os

class FootballAPI:
    def __init__(self):
        self.base_url = current_app.config['API_BASE_URL']
        self.api_key = self._load_api_key()
        
    def _load_api_key(self):
        """Carrega a chave da API do arquivo de configuração."""
        with open(current_app.config['API_KEY_FILE'], 'r') as f:
            return f.read().strip()
    
    def get_fixtures(self, league_id, season):
        """Obtém as partidas de uma liga específica."""
        url = f"{self.base_url}/fixtures/league/{league_id}"
        headers = {'X-RapidAPI-Key': self.api_key}
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
            
        return response.json()
    
    def get_fixture_statistics(self, fixture_id):
        """Obtém estatísticas de uma partida específica."""
        url = f"{self.base_url}/statistics/fixture/{fixture_id}"
        headers = {'X-RapidAPI-Key': self.api_key}
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
            
        return response.json()
    
    def save_fixture_stats(self, fixture_id, data, output_dir):
        """Salva as estatísticas da partida em um arquivo JSON."""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{fixture_id}.json")
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4) 