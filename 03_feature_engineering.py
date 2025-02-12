# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:23:13 2020

@author: mhayt
"""

print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import time
start=time.time()

import pickle
import logging
import os
from ml_functions.feature_engineering_functions import average_stats_df
from ml_functions.feature_engineering_functions import creating_ml_df

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#------------------------------- INPUT VARIABLES ------------------------------

# Find the stats dictionary file
try:
    dict_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') if f.endswith('stats_dict.txt')]
    if not dict_files:
        raise FileNotFoundError("No stats dictionary files found")
    stats_dict_saved_name = max(dict_files, key=lambda x: os.path.getctime(os.path.join('prem_clean_fixtures_and_dataframes', x)))
    logging.info(f"Using stats dictionary file: {stats_dict_saved_name}")
except Exception as e:
    logging.error(f"Error finding stats dictionary file: {str(e)}")
    raise

# Generate output filenames
df_5_output_name = 'prem_df_for_ml_5_v2.txt'
df_10_output_name = 'prem_df_for_ml_10_v2.txt'

#----------------------------- FEATURE ENGINEERING ----------------------------

try:
    # Load the game stats dictionary
    with open(f'prem_clean_fixtures_and_dataframes/{stats_dict_saved_name}', 'rb') as myFile:
        game_stats = pickle.load(myFile)
    logging.info("Successfully loaded game stats dictionary")

    # Create team list and sort
    team_list = list(game_stats.keys())
    team_list.sort()
    logging.info(f"Processing {len(team_list)} teams")

    # Create team fixture dictionary
    team_fixture_id_dict = {}
    for team in team_list:
        fix_id_list = sorted(game_stats[team].keys())
        team_fixture_id_dict[team] = fix_id_list
        logging.info(f"Team {team} has {len(fix_id_list)} fixtures")

    # Generate features with different sliding windows
    logging.info("Generating 5-game sliding window features")
    df_ml_5 = average_stats_df(5, team_list, team_fixture_id_dict, game_stats)
    
    logging.info("Generating 10-game sliding window features")
    df_ml_10 = average_stats_df(10, team_list, team_fixture_id_dict, game_stats)

    # Verificar se os DataFrames têm dados antes de salvar
    if df_ml_5 is not None and len(df_ml_5) > 0:
        logging.info(f"5-game window DataFrame shape: {df_ml_5.shape}")
        logging.info(f"5-game window columns: {df_ml_5.columns.tolist()}")
        logging.info(f"5-game window unique results: {df_ml_5['Result Indicator'].unique()}")
        
        # Create and save ML dataframes
        logging.info("Creating ML dataframe with 5-game window")
        df_for_ml_5_v2 = creating_ml_df(df_ml_5)
        with open(f'prem_clean_fixtures_and_dataframes/{df_5_output_name}', 'wb') as myFile:
            pickle.dump(df_for_ml_5_v2, myFile)
        logging.info(f"Saved 5-game window dataframe to {df_5_output_name}")
    else:
        logging.warning("5-game window DataFrame is empty or None")

    if df_ml_10 is not None and len(df_ml_10) > 0:
        logging.info(f"10-game window DataFrame shape: {df_ml_10.shape}")
        logging.info(f"10-game window columns: {df_ml_10.columns.tolist()}")
        logging.info(f"10-game window unique results: {df_ml_10['Result Indicator'].unique()}")
        
        logging.info("Creating ML dataframe with 10-game window")
        df_for_ml_10_v2 = creating_ml_df(df_ml_10)
        with open(f'prem_clean_fixtures_and_dataframes/{df_10_output_name}', 'wb') as myFile:
            pickle.dump(df_for_ml_10_v2, myFile)
        logging.info(f"Saved 10-game window dataframe to {df_10_output_name}")
    else:
        logging.warning("10-game window DataFrame is empty or None")

    # Save CSV for Power BI
    if df_ml_10 is not None and len(df_ml_10) > 0:
        csv_output = 'prem_clean_fixtures_and_dataframes/df_for_powerbi.csv'
        df_for_ml_10_v2.to_csv(csv_output, index=False)
        logging.info(f"Saved Power BI CSV to {csv_output}")
    else:
        logging.warning("Could not save Power BI CSV - no data available")

except Exception as e:
    logging.error(f"Error during feature engineering: {str(e)}")
    raise

finally:
    print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
    print(' ----------------- END ----------------- \n')
