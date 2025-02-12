# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:31:13 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import time
start=time.time()

import pandas as pd
import math
import pickle
import os
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#------------------------------- INPUT VARIABLES ------------------------------

# Diretório de entrada e saída
input_dir = 'prem_clean_fixtures_and_dataframes'
output_dir = 'prem_clean_fixtures_and_dataframes'
json_dir = 'prem_game_stats_json_files'

# Encontrar o arquivo CSV mais recente
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado no diretório.")

latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))
fixtures_file = os.path.join(input_dir, latest_csv)

# Nome do arquivo de saída
output_file = 'prem_all_stats_dict.txt'

#---------------------------- CREATING DF PER TEAM ----------------------------

#in this section we will create a nested dictionary containing the 20 teams, each with a value as another dictionary. In this dictionary we will have the game id along with the game dataframe.

fixtures_clean = pd.read_csv(fixtures_file)

#creating the 'fixtures_clean' ID index which we will use to take data from this dataframe and add to each of our individual fixture stats dataframe.
fixtures_clean_ID_index = pd.Index(fixtures_clean['Fixture ID'])

#team id list that we can iterate over
team_list = pd.concat([fixtures_clean['Home Team'], fixtures_clean['Away Team']]).unique().tolist()

#creating our dictionary which we will populate with data
all_stats_dict = {}

def process_statistics(stats_data):
    """Convert API statistics format to DataFrame format"""
    processed_stats = {}
    
    # Initialize default values
    default_stats = {
        'Shots on Goal': 0,
        'Total Shots': 0,
        'Shots insidebox': 0,
        'Shots outsidebox': 0,
        'Fouls': 0,
        'Corner Kicks': 0,
        'Offsides': 0,
        'Ball Possession': '0%',
        'Yellow Cards': 0,
        'Red Cards': 0,
        'Goalkeeper Saves': 0,
        'Total passes': 0,
        'Passes accurate': 0,
        'Passes %': '0%'
    }
    
    # Verificar se temos dados válidos
    if not stats_data or 'response' not in stats_data or len(stats_data['response']) < 2:
        # Retornar DataFrame com valores padrão se não houver dados
        return pd.DataFrame({stat: [default_value, default_value] for stat, default_value in default_stats.items()})
    
    # Process home team stats (primeiro time na resposta)
    home_stats = {stat['type']: stat['value'] for stat in stats_data['response'][0]['statistics']} if stats_data['response'] else {}
    away_stats = {stat['type']: stat['value'] for stat in stats_data['response'][1]['statistics']} if len(stats_data['response']) > 1 else {}
    
    # Create DataFrame with both home and away stats
    for stat_name, default_value in default_stats.items():
        home_value = home_stats.get(stat_name, default_value)
        away_value = away_stats.get(stat_name, default_value)
        
        # Convert None or null to default values
        home_value = default_value if home_value is None else home_value
        away_value = default_value if away_value is None else away_value
        
        # Handle percentage values
        if isinstance(home_value, str) and '%' in home_value:
            home_value = home_value.replace('%', '')
        if isinstance(away_value, str) and '%' in away_value:
            away_value = away_value.replace('%', '')
            
        processed_stats[stat_name] = [home_value, away_value]
    
    return pd.DataFrame(processed_stats)

#nested for loop to create nested dictionary, first key by team id, second key by fixture id
for team in team_list:
    logging.info(f'Processing team: {team}')
    
    #working the home teams
    team_fixture_list = []
    for i in fixtures_clean.index[:]:
        if fixtures_clean['Home Team'].iloc[i] == team:
            if not pd.isna(fixtures_clean['Home Goals'].iloc[i]):
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    
    all_stats_dict[team] = {}
    for j in team_fixture_list:
        try:
            #loading stats
            with open(f'prem_game_stats_json_files/{j}.json', 'r') as f:
                stats_data = json.load(f)
            
            #convert to dataframe
            df = process_statistics(stats_data)
            
            #adding home vs away goals to df
            temp_index = fixtures_clean_ID_index.get_loc(j)
            home_goals = fixtures_clean['Home Goals'].iloc[temp_index]
            away_goals = fixtures_clean['Away Goals'].iloc[temp_index]
            df['Goals'] = [home_goals, away_goals]
            
            #adding points data
            if home_goals > away_goals:
                df['Points'] = [2,0]
            elif home_goals == away_goals:
                df['Points'] = [1,1]
            elif home_goals < away_goals:
                df['Points'] = [0,2]
            else:
                df['Points'] = [0, 0]
            
            #adding home-away identifier to df
            df['Team Identifier'] = [1,2]
            
            #adding team id and team name
            df['Team ID'] = [fixtures_clean['Home Team ID'].iloc[temp_index], fixtures_clean['Away Team ID'].iloc[temp_index]]
            df['Team'] = [fixtures_clean['Home Team'].iloc[temp_index], fixtures_clean['Away Team'].iloc[temp_index]]
            
            #adding game date
            gd = fixtures_clean['Date'].iloc[temp_index]
            df['Game Date'] = [gd, gd]
            
            #filling NA values with 0
            df = df.fillna(0)
            
            #adding this modified df to nested dictionary
            all_stats_dict[team][j] = df
            
        except Exception as e:
            logging.error(f'Error processing fixture {j} for team {team}: {str(e)}')
            continue
    
    #working the away teams    
    team_fixture_list = []    
    for i in fixtures_clean.index[:]:
        if fixtures_clean['Away Team'].iloc[i] == team:
            if not pd.isna(fixtures_clean['Away Goals'].iloc[i]):
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    
    for j in team_fixture_list:
        try:
            #loading stats
            with open(f'prem_game_stats_json_files/{j}.json', 'r') as f:
                stats_data = json.load(f)
            
            #convert to dataframe
            df = process_statistics(stats_data)
            
            #adding home vs away goals to df
            temp_index = fixtures_clean_ID_index.get_loc(j)
            home_goals = fixtures_clean['Home Goals'].iloc[temp_index]
            away_goals = fixtures_clean['Away Goals'].iloc[temp_index]
            df['Goals'] = [home_goals, away_goals]
            
            #adding points data
            if home_goals > away_goals:
                df['Points'] = [2,0]
            elif home_goals == away_goals:
                df['Points'] = [1,1]
            elif home_goals < away_goals:
                df['Points'] = [0,2]
            else:
                df['Points'] = [0, 0]
            
            #adding home-away identifier to df
            df['Team Identifier'] = [2,1]
            
            #adding team id and team name
            df['Team ID'] = [fixtures_clean['Home Team ID'].iloc[temp_index], fixtures_clean['Away Team ID'].iloc[temp_index]]
            df['Team'] = [fixtures_clean['Home Team'].iloc[temp_index], fixtures_clean['Away Team'].iloc[temp_index]]
            
            #adding game date
            gd = fixtures_clean['Date'].iloc[temp_index]
            df['Game Date'] = [gd, gd]
            
            #filling NA values with 0
            df = df.fillna(0)
            
            #adding this modified df to nested dictionary
            all_stats_dict[team][j] = df
            
        except Exception as e:
            logging.error(f'Error processing fixture {j} for team {team}: {str(e)}')
            continue

#saving our generated dictionary as a pickle file
output_path = os.path.join(output_dir, output_file)
with open(output_path, 'wb') as myFile:
    pickle.dump(all_stats_dict, myFile)

logging.info(f'Successfully saved statistics dictionary to {output_path}')

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
