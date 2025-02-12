# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:45:42 2020

@author: mhayt
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import joblib
from ml_functions.feature_engineering_functions import average_stats_df, mod_df

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictions/prediction_logs.log'),
        logging.StreamHandler()
    ]
)

print('\n\n ---------------- START ---------------- \n')

class PremierLeaguePredictor:
    """Classe para gerenciar predições de jogos da Premier League"""
    
    def __init__(self):
        self.fixtures_clean = None
        self.game_stats = None
        self.team_list = []
        self.team_fixture_id_dict = {}
        self.unplayed_games = None
        self.df_for_predictions = None
        self.predictions = None
        self.models = {}
        
        # Criar diretórios necessários
        os.makedirs('predictions/results', exist_ok=True)
        os.makedirs('predictions/figures', exist_ok=True)
        
    def load_data(self) -> None:
        """Carrega dados necessários para predições"""
        try:
            # Carregar dados de fixtures
            fixtures_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') 
                            if f.endswith('fixtures_df.csv')]
            latest_fixtures = max(fixtures_files, key=lambda x: os.path.getctime(
                os.path.join('prem_clean_fixtures_and_dataframes', x)))
            
            self.fixtures_clean = pd.read_csv(f'prem_clean_fixtures_and_dataframes/{latest_fixtures}')
            logging.info(f"Fixtures carregados: {latest_fixtures}")
            
            # Carregar estatísticas dos jogos
            stats_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') 
                          if f.endswith('stats_dict.txt')]
            latest_stats = max(stats_files, key=lambda x: os.path.getctime(
                os.path.join('prem_clean_fixtures_and_dataframes', x)))
            
            with open(f'prem_clean_fixtures_and_dataframes/{latest_stats}', 'rb') as f:
                self.game_stats = pickle.load(f)
            logging.info(f"Estatísticas carregadas: {latest_stats}")
            
            # Carregar modelos
            self._load_models()
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {str(e)}")
            raise
            
    def _load_models(self) -> None:
        """Carrega os modelos treinados"""
        model_dirs = [
            'ml_model_build_random_forest/ml_models',
            'ml_model_build_xgboost/ml_models',
            'ml_model_build_neural_network/ml_models',
            'ml_model_build_ensemble_model/ml_models'
        ]
        
        for dir_path in model_dirs:
            if os.path.exists(dir_path):
                model_files = [f for f in os.listdir(dir_path) 
                             if f.endswith('.pkl') and not f.startswith('scaler') 
                             and not f.startswith('label')]
                
                for model_file in model_files:
                    model_name = model_file.split('_')[0]
                    model_path = os.path.join(dir_path, model_file)
                    
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logging.info(f"Modelo carregado: {model_file}")
                    except Exception as e:
                        logging.warning(f"Erro ao carregar modelo {model_file}: {str(e)}")
    
    def prepare_team_data(self) -> None:
        """Prepara dados dos times para predições"""
        try:
            # Criar lista de times
            self.team_list = sorted(list(self.game_stats.keys()))
            
            # Criar dicionário de IDs de jogos por time
            for team in self.team_list:
                fix_id_list = sorted(list(self.game_stats[team].keys()))
                self.team_fixture_id_dict[team] = fix_id_list
            
            # Criar versão reduzida com apenas os últimos 10 jogos
            team_fixture_id_dict_reduced = {
                team: self.team_fixture_id_dict[team][-10:]
                for team in self.team_fixture_id_dict
            }
            
            # Preparar dados para predições
            df_10_upcom_fix_e = average_stats_df(
                10, self.team_list, team_fixture_id_dict_reduced, 
                self.game_stats, making_predictions=True
            )
            self.df_for_predictions = mod_df(df_10_upcom_fix_e, making_predictions=True)
            
            logging.info("Dados dos times preparados com sucesso")
            
        except Exception as e:
            logging.error(f"Erro ao preparar dados dos times: {str(e)}")
            raise
    
    def get_unplayed_games(self) -> None:
        """Identifica jogos não realizados"""
        try:
            # Identificar jogos já realizados
            played_games = self.fixtures_clean[
                self.fixtures_clean['Home Team Goals'].notna()
            ].index.tolist()
            
            # Filtrar jogos não realizados
            self.unplayed_games = self.fixtures_clean.drop(played_games).reset_index(drop=True)
            self.unplayed_games = self.unplayed_games.drop(
                ['Home Team Goals', 'Away Team Goals'], axis=1
            )
            
            logging.info(f"Identificados {len(self.unplayed_games)} jogos não realizados")
            
        except Exception as e:
            logging.error(f"Erro ao identificar jogos não realizados: {str(e)}")
            raise
    
    def model_missing_data(self) -> pd.DataFrame:
        """Modela dados faltantes para times recém-promovidos"""
        try:
            # IDs dos times rebaixados (atualizar conforme necessário)
            relegated_teams = [35, 38, 71]  # Atualizar com IDs corretos
            
            # Calcular média dos times rebaixados
            relegated_dfs = []
            for team_id in relegated_teams:
                team_df = self.df_for_predictions[
                    self.df_for_predictions['Team ID'] == team_id
                ].reset_index(drop=True)
                if not team_df.empty:
                    relegated_dfs.append(team_df)
            
            if relegated_dfs:
                average_df = pd.concat(relegated_dfs).groupby(level=0).mean()
                logging.info("Dados modelados para times recém-promovidos")
                return average_df
            else:
                logging.warning("Sem dados de times rebaixados disponíveis")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Erro ao modelar dados faltantes: {str(e)}")
            raise
    
    def prepare_prediction_data(self) -> None:
        """Prepara dados para predições"""
        try:
            # Criar DataFrame base para predições
            columns = self.df_for_predictions.columns.tolist()[:14]
            self.df_for_predictions = pd.DataFrame(
                np.zeros((len(self.unplayed_games), 14)),
                columns=columns
            )
            
            # Adicionar informações dos times
            self.df_for_predictions['Home Team ID'] = self.unplayed_games['Home Team ID']
            self.df_for_predictions['Away Team ID'] = self.unplayed_games['Away Team ID']
            self.df_for_predictions['Home Team'] = self.unplayed_games['Home Team']
            self.df_for_predictions['Away Team'] = self.unplayed_games['Away Team']
            self.df_for_predictions['Game Date'] = self.unplayed_games['Game Date']
            
            # Modelar dados faltantes
            average_stats = self.model_missing_data()
            
            # Preencher estatísticas
            team_ids = list(self.df_for_predictions['Team ID'])
            for i in range(len(self.unplayed_games)):
                home_team = self.unplayed_games['Home Team ID'].iloc[i]
                away_team = self.unplayed_games['Away Team ID'].iloc[i]
                
                # Obter estatísticas dos times
                home_stats = (self.df_for_predictions[
                    self.df_for_predictions['Team ID'] == home_team
                ] if home_team in team_ids else average_stats).iloc[0, :7].values
                
                away_stats = (self.df_for_predictions[
                    self.df_for_predictions['Team ID'] == away_team
                ] if away_team in team_ids else average_stats).iloc[0, :7].values
                
                # Preencher estatísticas
                self.df_for_predictions.iloc[i, 0:7] = home_stats
                self.df_for_predictions.iloc[i, 7:14] = away_stats
            
            logging.info("Dados preparados para predições")
            
        except Exception as e:
            logging.error(f"Erro ao preparar dados para predições: {str(e)}")
            raise
    
    def make_predictions(self) -> None:
        """Realiza predições usando os modelos carregados"""
        try:
            predictions_all = {}
            df_features = self.df_for_predictions.drop(
                ['Home Team ID', 'Away Team ID', 'Home Team', 'Away Team', 'Game Date'],
                axis=1
            )
            
            # Fazer predições com cada modelo
            for model_name, model in self.models.items():
                predictions_raw = model.predict_proba(df_features)
                predictions_df = pd.DataFrame(
                    data=predictions_raw * 100,
                    columns=['Away Win', 'Draw', 'Home Win']
                ).round(1)
                predictions_all[model_name] = predictions_df
            
            # Calcular média das predições (ensemble)
            predictions_combined = pd.concat(predictions_all.values()).groupby(level=0).mean()
            
            # Criar DataFrame final
            self.predictions = pd.concat(
                [self.unplayed_games, predictions_combined],
                axis=1,
                join='inner'
            )
            
            # Reordenar colunas
            columns = [
                'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win',
                'Game Date', 'Venue', 'Home Team Logo', 'Away Team Logo',
                'Home Team ID', 'Away Team ID', 'Fixture ID', 'index'
            ]
            self.predictions = self.predictions.reindex(columns=columns)
            
            logging.info("Predições realizadas com sucesso")
            
        except Exception as e:
            logging.error(f"Erro ao realizar predições: {str(e)}")
            raise
    
    def plot_predictions(self) -> None:
        """Gera visualizações das predições"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Plot de probabilidades por jogo
            plt.figure(figsize=(15, 8))
            games = [f"{row['Home Team']} vs {row['Away Team']}" 
                    for _, row in self.predictions.iterrows()]
            
            data = {
                'Home Win': self.predictions['Home Win'],
                'Draw': self.predictions['Draw'],
                'Away Win': self.predictions['Away Win']
            }
            
            df_plot = pd.DataFrame(data, index=games)
            ax = df_plot.plot(kind='bar', stacked=True)
            plt.title('Probabilidades de Resultado por Jogo')
            plt.xlabel('Jogos')
            plt.ylabel('Probabilidade (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'predictions/figures/probabilities_{timestamp}.png')
            plt.close()
            
            logging.info("Visualizações geradas com sucesso")
            
        except Exception as e:
            logging.error(f"Erro ao gerar visualizações: {str(e)}")
            raise
    
    def save_predictions(self) -> None:
        """Salva as predições geradas"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Salvar em formato CSV
            csv_path = f'predictions/results/predictions_{timestamp}.csv'
            self.predictions.to_csv(csv_path, index=False)
            
            # Salvar em formato pickle
            pickle_path = f'predictions/results/predictions_{timestamp}.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.predictions, f)
            
            # Salvar cópia para o servidor web
            web_path = 'web_server/pl_predictions.csv'
            if os.path.exists('web_server'):
                self.predictions.to_csv(web_path, index=False)
            
            logging.info(f"Predições salvas em {csv_path} e {pickle_path}")
            
        except Exception as e:
            logging.error(f"Erro ao salvar predições: {str(e)}")
            raise
    
    def run_prediction_pipeline(self) -> None:
        """Executa o pipeline completo de predições"""
        try:
            logging.info("Iniciando pipeline de predições...")
            
            self.load_data()
            self.prepare_team_data()
            self.get_unplayed_games()
            self.prepare_prediction_data()
            self.make_predictions()
            self.plot_predictions()
            self.save_predictions()
            
            logging.info("Pipeline de predições concluído com sucesso")
            
        except Exception as e:
            logging.error(f"Erro no pipeline de predições: {str(e)}")
            raise

def main():
    try:
        predictor = PremierLeaguePredictor()
        predictor.run_prediction_pipeline()
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main()
