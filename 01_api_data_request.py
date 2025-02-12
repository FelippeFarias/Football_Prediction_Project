# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:33:10 2020

@author: mhayt
"""

import requests
import json
import logging
import pandas as pd
import os
import math
import time
from os import listdir
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()
    ]
)

# API Configuration
BASE_URL = 'https://v3.football.api-sports.io'
try:
    API_KEY = open('api_key.txt', mode='r').read().strip()
except FileNotFoundError:
    logging.error("Arquivo api_key.txt não encontrado")
    raise
except Exception as e:
    logging.error(f"Erro ao ler api_key.txt: {str(e)}")
    raise

HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v3.football.api-sports.io'
}

# Define the variable to control missing data requests
request_missing_game_stats = True

# Helper functions
def get_api_data(url: str, headers: Dict[str, str] = HEADERS) -> Dict[str, Any]:
    """
    Faz requisição à API e retorna os dados
    
    Args:
        url: URL da requisição
        headers: Headers da requisição
    
    Returns:
        Dados da resposta em formato JSON
    
    Raises:
        RuntimeError: Se a requisição falhar
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f'Erro na requisição: {str(e)}')
        raise RuntimeError(f'Erro {response.status_code}') from e

def slice_api(api_data, start_char=0, end_char=0):
    return api_data['response']

def save_api_output(save_name: str, json_data: Dict[str, Any], json_data_path: str = '') -> None:
    """
    Salva dados da API em arquivo JSON
    
    Args:
        save_name: Nome do arquivo
        json_data: Dados para salvar
        json_data_path: Caminho do diretório
    """
    try:
        # Remover extensão .json se já estiver presente no save_name
        if save_name.endswith('.json'):
            save_name = save_name[:-5]
            
        # Construir caminho completo
        file_path = os.path.join(json_data_path, f'{save_name}.json')
        
        # Criar diretório pai se não existir
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Salvar arquivo
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        logging.info(f'Arquivo salvo em: {file_path}')
        
    except Exception as e:
        logging.error(f'Erro ao salvar arquivo {save_name}: {str(e)}')
        raise

def read_json_as_pd_df(json_data, json_data_path='', orient_def='records'):
    return pd.read_json(json_data_path + json_data, orient=orient_def)

def req_prem_stats_list(missing_data: List[int]) -> None:
    """
    Requisita estatísticas para uma lista de jogos
    
    Args:
        missing_data: Lista de IDs dos jogos faltantes
    """
    try:
        for fixture_id in missing_data:
            url = f'{BASE_URL}/fixtures/statistics?fixture={fixture_id}'
            logging.info(f'Requisitando estatísticas para fixture {fixture_id}')
            
            data = get_api_data(url)
            
            # Salvar arquivo
            save_path = os.path.join('prem_game_stats_json_files', str(fixture_id))
            save_api_output(save_path, data)
            
            # Aguardar para não sobrecarregar a API
            # time.sleep(0.5)
            
    except Exception as e:
        logging.error(f'Erro ao requisitar estatísticas: {str(e)}')
        raise

def get_all_seasons(league_id: int) -> List[int]:
    """
    Recupera todas as temporadas disponíveis para uma liga específica.
    
    Args:
        league_id: ID da liga
    
    Returns:
        Lista de temporadas disponíveis
    """
    url = f'{BASE_URL}/leagues/seasons'
    data = get_api_data(url)
    return data.get('response', [])

def main() -> None:
    """Função principal"""
    try:
        # Premier League ID is 39
        league_id = 39
        seasons = get_all_seasons(league_id)
        
        all_fixtures = []
        
        for season in seasons:
            # Buscar jogos de cada temporada
            url = f'{BASE_URL}/fixtures?league={league_id}&season={season}'
            logging.info(f'Requisitando dados de: {url}')
            
            data = get_api_data(url)
            logging.info(f'Número de resultados para a temporada {season}: {data.get("results", 0)}')
            
            if data.get('response'):
                all_fixtures.extend(data['response'])
        
        logging.info(f'Obtidos {len(all_fixtures)} fixtures no total')
        
        # Processar dados em DataFrame
        fixtures_df = pd.DataFrame([{
            'Fixture ID': fixture['fixture']['id'],
            'Date': fixture['fixture']['date'],
            'Venue': fixture['fixture']['venue']['name'],
            'Home Team ID': fixture['teams']['home']['id'],
            'Home Team': fixture['teams']['home']['name'],
            'Away Team ID': fixture['teams']['away']['id'],
            'Away Team': fixture['teams']['away']['name'],
            'Home Goals': fixture['goals']['home'],
            'Away Goals': fixture['goals']['away'],
            'Status': fixture['fixture']['status']['long'],
            'Home Team Logo': fixture['teams']['home']['logo'],
            'Away Team Logo': fixture['teams']['away']['logo']
        } for fixture in all_fixtures])
        
        # Criar diretórios
        os.makedirs('prem_clean_fixtures_and_dataframes', exist_ok=True)
        os.makedirs('prem_game_stats_json_files', exist_ok=True)
        
        # Salvar DataFrame
        output_path = 'prem_clean_fixtures_and_dataframes/premier_league_fixtures_2023.csv'
        fixtures_df.to_csv(output_path, index=False)
        logging.info(f'DataFrame salvo em {output_path}')
        
        # Verificar dados faltantes
        try:
            existing_data = [int(i[:-5]) for i in os.listdir('prem_game_stats_json_files/') if i.endswith('.json')]
        except FileNotFoundError:
            existing_data = []
        
        missing_data = []
        for i in fixtures_df.index:
            fix_id = fixtures_df['Fixture ID'].iloc[i]
            if fix_id not in existing_data and not pd.isna(fixtures_df['Home Goals'].iloc[i]):
                missing_data.append(fix_id)
        
        if missing_data:
            logging.info(f'Encontrados {len(missing_data)} fixtures faltantes')
            req_prem_stats_list(missing_data)
        else:
            logging.info('Nenhum fixture faltante encontrado')
            
    except Exception as e:
        logging.error(f'Erro na execução: {str(e)}')
        raise

if __name__ == '__main__':
    print('\n\n ---------------- START ---------------- \n')
    start_time = time.time()
    
    try:
        main()
    except Exception as e:
        logging.error(f'Erro fatal: {str(e)}')
    finally:
        elapsed_time = time.time() - start_time
        print(f'\nTempo de execução: {elapsed_time/60:.2f} minutos')
        print('\n ---------------- END ---------------- \n')
