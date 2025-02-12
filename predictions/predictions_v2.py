# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:57:52 2020
@author: mhayt
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import joblib

# Definir diretório raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Adicionar diretório raiz ao PYTHONPATH
sys.path.append(PROJECT_ROOT)

# Importar funções do módulo ml_functions
from ml_functions.feature_engineering_functions import average_stats_df, mod_df

# Criar diretório de logs se não existir
LOGS_DIR = os.path.join(PROJECT_ROOT, 'predictions', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'prediction_logs.log')),
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
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, '..'))
        
        dirs_to_create = [
            os.path.join(self.base_dir, 'results'),
            os.path.join(self.base_dir, 'figures'),
            os.path.join(self.base_dir, 'logs'),
            os.path.join(self.project_root, 'prem_clean_fixtures_and_dataframes'),
            os.path.join(self.project_root, 'results')
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        
    def load_data(self) -> None:
        """Carrega dados necessários para predições"""
        try:
            # Carregar dados de fixtures
            fixtures_files = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes')) 
                            if 'fixtures' in f.lower() and f.endswith('.csv')]
            latest_fixtures = max(fixtures_files, key=lambda x: os.path.getctime(
                os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', x)))
            
            self.fixtures_clean = pd.read_csv(os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', latest_fixtures))
            
            # Verificar se há jogos futuros
            unplayed_games = self.fixtures_clean[self.fixtures_clean['Status'] != 'Match Finished']
            if len(unplayed_games) == 0:
                logging.warning("Não há jogos futuros no arquivo de fixtures. Por favor, atualize os fixtures usando o script de coleta de dados.")
                raise ValueError("Fixtures precisam ser atualizados")
            
            logging.info(f"Fixtures carregados: {latest_fixtures}")
            logging.info(f"Encontrados {len(unplayed_games)} jogos futuros")
            
            # Carregar estatísticas dos jogos
            stats_files = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes')) 
                          if f.endswith('stats_dict.txt')]
            latest_stats = max(stats_files, key=lambda x: os.path.getctime(
                os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', x)))
            
            with open(os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', latest_stats), 'rb') as f:
                self.game_stats = pickle.load(f)
            logging.info(f"Estatísticas carregadas: {latest_stats}")
            
            # Carregar modelos
            self.load_models()
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {str(e)}")
            raise
            
    def load_models(self) -> None:
        """Carrega modelos salvos"""
        try:
            model_dirs = [
                os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest/ml_models'),
                os.path.join(PROJECT_ROOT, 'ml_model_build_xgboost/ml_models')
            ]
            
            for dir_path in model_dirs:
                if os.path.exists(dir_path):
                    # Listar todos os arquivos .pkl no diretório
                    model_files = [f for f in os.listdir(dir_path) 
                                 if f.endswith('.pkl') and 'model' in f.lower()
                                 and not f.startswith('scaler') 
                                 and not f.startswith('label')]
                    
                    # Ordenar por data de criação (mais recente primeiro)
                    model_files.sort(key=lambda x: os.path.getctime(os.path.join(dir_path, x)), reverse=True)
                    
                    # Pegar apenas os modelos mais recentes (um para janela 5 e um para janela 10)
                    latest_models = []
                    for window in ['5', '10']:
                        for model in model_files:
                            if f'_{window}_' in model:
                                latest_models.append(model)
                                break
                    
                    # Carregar os modelos
                    for model_file in latest_models:
                        model_path = os.path.join(dir_path, model_file)
                        try:
                            model = joblib.load(model_path)
                            if hasattr(model, 'predict_proba'):
                                # Extrair nome do modelo (random_forest ou xgboost) e janela (5 ou 10)
                                model_type = 'random_forest' if 'random_forest' in model_file else 'xgboost'
                                window = '5' if '_5_' in model_file else '10'
                                model_name = f'{model_type}_{window}'
                                self.models[model_name] = model
                                logging.info(f"Modelo carregado: {model_file}")
                            else:
                                logging.warning(f"Arquivo {model_file} não é um modelo válido")
                        except Exception as e:
                            logging.warning(f"Erro ao carregar modelo {model_file}: {str(e)}")
                            
            if not self.models:
                raise ValueError("Nenhum modelo válido foi carregado")
                
        except Exception as e:
            logging.error(f"Erro ao carregar modelos: {str(e)}")
            raise
    
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
                self.game_stats)
            self.df_for_predictions = mod_df(df_10_upcom_fix_e, window_size=10)
            
            logging.info("Dados dos times preparados com sucesso")
            
        except Exception as e:
            logging.error(f"Erro ao preparar dados dos times: {str(e)}")
            raise
    
    def get_unplayed_games(self) -> None:
        """Identifica jogos não realizados"""
        try:
            # Identificar jogos já realizados
            played_games = self.fixtures_clean[
                self.fixtures_clean['Home Goals'].notna()
            ].index.tolist()
            
            # Filtrar jogos não realizados
            self.unplayed_games = self.fixtures_clean.drop(played_games).reset_index(drop=True)
            self.unplayed_games = self.unplayed_games.drop(
                ['Home Goals', 'Away Goals'], axis=1
            )
            
            # Renomear coluna Date para Game Date se necessário
            if 'Date' in self.unplayed_games.columns and 'Game Date' not in self.unplayed_games.columns:
                self.unplayed_games = self.unplayed_games.rename(columns={'Date': 'Game Date'})
            
            if len(self.unplayed_games) == 0:
                logging.warning("Não há jogos futuros para prever")
                raise ValueError("Não há jogos futuros para prever")
            
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
                # Verificar tanto Home Team ID quanto Away Team ID
                home_team_data = self.df_for_predictions[
                    self.df_for_predictions['Home Team ID'] == team_id
                ].reset_index(drop=True)
                
                away_team_data = self.df_for_predictions[
                    self.df_for_predictions['Away Team ID'] == team_id
                ].reset_index(drop=True)
                
                if not home_team_data.empty:
                    relegated_dfs.append(home_team_data)
                if not away_team_data.empty:
                    relegated_dfs.append(away_team_data)
            
            if relegated_dfs:
                # Calcular média apenas das colunas numéricas
                numeric_columns = self.df_for_predictions.select_dtypes(include=['float64', 'int64']).columns
                average_df = pd.concat(relegated_dfs)[numeric_columns].mean().to_frame().T
                logging.info("Dados modelados para times recém-promovidos")
                return average_df
            else:
                logging.warning("Sem dados de times rebaixados disponíveis")
                # Retornar DataFrame vazio com as mesmas colunas numéricas
                numeric_columns = self.df_for_predictions.select_dtypes(include=['float64', 'int64']).columns
                return pd.DataFrame(columns=numeric_columns)
                
        except Exception as e:
            logging.error(f"Erro ao modelar dados faltantes: {str(e)}")
            raise
    
    def _prepare_features(self):
        """Prepara as features para predições"""
        try:
            # Lista de estatísticas base para o RandomForest (8 estatísticas base)
            rf_base_stats = [
                'Shots on Goal', 'Total Shots', 'Shots Inside Box',
                'Fouls', 'Corner Kicks', 'Ball Possession',
                'Yellow Cards', 'Goalkeeper Saves'
            ]
            
            # Features para XGBoost na ordem correta
            xgb_features = [
                'Team Av Shots Diff',
                'Team Av Shots Inside Box Diff',
                'Team Av Fouls Diff',
                'Team Av Corners Diff',
                'Team Av Possession Diff',
                'Team Av Pass Accuracy Diff',
                'Team Av Goal Diff',
                'Opponent Av Shots Diff',
                'Opponent Av Shots Inside Box Diff',
                'Opponent Av Fouls Diff',
                'Opponent Av Corners Diff',
                'Opponent Av Possession Diff',
                'Opponent Av Pass Accuracy Diff',
                'Opponent Av Goal Diff'
            ]
            
            # Mapeamento para XGBoost
            xgb_mapping = {
                'Total Shots': 'Shots',
                'Shots Inside Box': 'Shots Inside Box',
                'Fouls': 'Fouls',
                'Corner Kicks': 'Corners',
                'Ball Possession': 'Possession',
                'Pass Accuracy': 'Pass Accuracy',
                'Goals': 'Goal'
            }
            
            # Features para RandomForest
            rf_features = []
            for stat in rf_base_stats:
                # Features do time da casa
                rf_features.extend([
                    f'Team Av {stat} Mean',
                    f'Team Av {stat} Diff'
                ])
                # Features do time visitante
                rf_features.extend([
                    f'Opponent Av {stat} Mean',
                    f'Opponent Av {stat} Diff'
                ])
            # Adicionar feature de gols
            rf_features.append('Team Av Goals Mean')
            
            # Criar DataFrames separados para cada modelo
            self.rf_predictions = pd.DataFrame(columns=rf_features)
            self.xgb_predictions = pd.DataFrame(columns=xgb_features)
            
            # Preencher os DataFrames com os dados
            for idx, row in self.unplayed_games.iterrows():
                home_team = row['Home Team']
                away_team = row['Away Team']
                
                # Calcular estatísticas para time da casa
                home_stats = self._calculate_team_stats(home_team, list(set(rf_base_stats + ['Goals', 'Pass Accuracy'])))
                # Calcular estatísticas para time visitante
                away_stats = self._calculate_team_stats(away_team, list(set(rf_base_stats + ['Goals', 'Pass Accuracy'])))
                
                # Features para RandomForest
                rf_game_features = {}
                for stat in rf_base_stats:
                    # Features do time da casa
                    rf_game_features[f'Team Av {stat} Mean'] = home_stats.get(f'{stat} Mean', 0)
                    rf_game_features[f'Team Av {stat} Diff'] = home_stats.get(f'{stat} Diff', 0)
                    # Features do time visitante
                    rf_game_features[f'Opponent Av {stat} Mean'] = away_stats.get(f'{stat} Mean', 0)
                    rf_game_features[f'Opponent Av {stat} Diff'] = away_stats.get(f'{stat} Diff', 0)
                # Adicionar média de gols
                rf_game_features['Team Av Goals Mean'] = home_stats.get('Goals Mean', 0)
                
                # Features para XGBoost
                xgb_game_features = {}
                for feature in xgb_features:
                    # Mapear nomes de features para estatísticas originais
                    for orig_stat, mapped_stat in xgb_mapping.items():
                        if mapped_stat in feature:
                            orig_feature = feature.replace(mapped_stat, orig_stat)
                            if 'Team Av' in feature:
                                xgb_game_features[feature] = home_stats.get(orig_feature.replace('Team Av ', ''), 0)
                            else:  # Opponent Av
                                xgb_game_features[feature] = away_stats.get(orig_feature.replace('Opponent Av ', ''), 0)
                            break
                    else:
                        # Se não encontrou mapeamento, usar o valor diretamente
                        if 'Team Av' in feature:
                            xgb_game_features[feature] = home_stats.get(feature.replace('Team Av ', ''), 0)
                        else:  # Opponent Av
                            xgb_game_features[feature] = away_stats.get(feature.replace('Opponent Av ', ''), 0)
                
                # Adicionar linhas aos DataFrames
                self.rf_predictions.loc[idx] = pd.Series(rf_game_features)
                self.xgb_predictions.loc[idx] = pd.Series(xgb_game_features)
            
            # Preencher valores NaN com 0
            self.rf_predictions = self.rf_predictions.fillna(0)
            self.xgb_predictions = self.xgb_predictions.fillna(0)
            
            # Log das features geradas
            logging.info(f"\nFeatures RandomForest ({len(rf_features)}):")
            for col in sorted(rf_features):
                logging.info(f"- {col}")
                
            logging.info(f"\nFeatures XGBoost ({len(xgb_features)}):")
            for col in xgb_features:  # Manter ordem original
                logging.info(f"- {col}")
            
            logging.info("Dados preparados para predições dos modelos.")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao preparar features: {str(e)}")
            return False
    
    def _calculate_team_stats(self, team: str, base_stats: List[str]) -> Dict[str, float]:
        """Calcula estatísticas para um time específico"""
        try:
            team_stats = {}
            team_data = []
            
            # Coletar dados históricos do time
            if team in self.game_stats:
                for fixture_id, game_data in self.game_stats[team].items():
                    # Identificar se é time da casa ou visitante
                    team_idx = 0 if game_data['Team'].iloc[0] == team else 1
                    
                    # Coletar estatísticas do jogo
                    game_stats = {}
                    for stat in base_stats:
                        if stat in game_data.columns:
                            try:
                                value = float(game_data[stat].iloc[team_idx])
                                game_stats[stat] = value
                            except (ValueError, TypeError):
                                game_stats[stat] = 0
                    team_data.append(game_stats)
            
            # Calcular médias e diferenças
            if team_data:
                for stat in base_stats:
                    values = [game[stat] for game in team_data if stat in game]
                    if values:
                        mean_value = np.mean(values)
                        last_value = values[-1] if values else 0
                        team_stats[f'{stat} Mean'] = mean_value
                        team_stats[f'{stat} Diff'] = last_value - mean_value
                    else:
                        team_stats[f'{stat} Mean'] = 0
                        team_stats[f'{stat} Diff'] = 0
            else:
                # Se não houver dados, usar zeros
                for stat in base_stats:
                    team_stats[f'{stat} Mean'] = 0
                    team_stats[f'{stat} Diff'] = 0
            
            return team_stats
        except Exception as e:
            logging.error(f"Erro ao calcular estatísticas para {team}: {str(e)}")
            # Criar dicionário de zeros para todas as features
            default_stats = {}
            for stat in base_stats:
                default_stats[f'{stat} Mean'] = 0
                default_stats[f'{stat} Diff'] = 0
            return default_stats
            
    def _calculate_team_fatigue(self, team_id: str, window: int = 30) -> float:
        """Calcula a fadiga do time baseado em jogos recentes"""
        try:
            if team_id not in self.game_stats:
                return 0.5
                
            recent_games = list(self.game_stats[team_id].values())[-5:]
            if not recent_games:
                return 0.5
                
            # Calcular média de dias entre jogos
            dates = []
            for game in recent_games:
                if 'Game Date' not in game.columns:
                    continue
                dates.append(pd.to_datetime(game['Game Date'].iloc[0]))
            
            if len(dates) < 2:
                return 0.5
                
            dates.sort()
            days_between = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_days = np.mean(days_between)
            
            # Normalizar fadiga (mais dias = menos fadiga)
            fatigue = 1 - (avg_days / window)
            return np.clip(fatigue, 0, 1)
            
        except Exception:
            return 0.5
            
    def _calculate_team_form(self, team: str, window: int = 5) -> float:
        """Calcula a forma do time baseado em resultados recentes"""
        try:
            if team not in self.game_stats:
                return 0.5
                
            recent_games = list(self.game_stats[team].values())[-window:]
            if not recent_games:
                return 0.5
                
            results = []
            for game in recent_games:
                if 'Goals' not in game.columns:
                    continue
                    
                team_idx = 0 if game['Team'].iloc[0] == team else 1
                team_goals = float(game['Goals'].iloc[team_idx])
                opponent_goals = float(game['Goals'].iloc[1 - team_idx])
                
                if team_goals > opponent_goals:
                    results.append(1)  # Vitória
                elif team_goals == opponent_goals:
                    results.append(0.5)  # Empate
                else:
                    results.append(0)  # Derrota
            
            if not results:
                return 0.5
                
            # Calcular média ponderada dando mais peso aos jogos mais recentes
            weights = [1 + (i/len(results)) for i in range(len(results))]
            weighted_form = sum(r * w for r, w in zip(results, weights)) / sum(weights)
            
            return weighted_form
            
        except Exception as e:
            logging.error(f"Erro ao calcular forma para {team}: {str(e)}")
            return 0.5
    
    def _calculate_team_momentum(self, team: str, window: int = 5) -> float:
        """Calcula o momentum do time baseado em resultados recentes"""
        try:
            if team not in self.game_stats:
                return 0.5
                
            recent_games = list(self.game_stats[team].values())[-window:]
            if not recent_games:
                return 0.5
                
            points = []
            for game in recent_games:
                if 'Goals' not in game.columns:
                    continue
                    
                team_idx = 0 if game['Team'].iloc[0] == team else 1
                team_goals = float(game['Goals'].iloc[team_idx])
                opponent_goals = float(game['Goals'].iloc[1 - team_idx])
                
                if team_goals > opponent_goals:
                    points.append(3)
                elif team_goals == opponent_goals:
                    points.append(1)
                else:
                    points.append(0)
            
            if not points:
                return 0.5
                
            momentum = sum(points) / (len(points) * 3)  # Normalizado entre 0 e 1
            return momentum
            
        except Exception as e:
            logging.error(f"Erro ao calcular momentum para {team}: {str(e)}")
            return 0.5
    
    def make_predictions(self) -> None:
        """Realiza predições usando os modelos carregados"""
        try:
            predictions_all = {}
            
            # Fazer predições com cada modelo
            for model_name, model in self.models.items():
                # Selecionar DataFrame apropriado
                if 'random_forest' in model_name:
                    df_features = self.rf_predictions
                else:  # xgboost
                    df_features = self.xgb_predictions
                
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
            
            # Adicionar previsões de estatísticas adicionais
            for idx, row in self.predictions.iterrows():
                home_team = row['Home Team']
                away_team = row['Away Team']
                
                # Calcular médias das últimas partidas
                home_stats = self._calculate_team_stats(home_team, ['Fouls', 'Yellow Cards', 'Corner Kicks', 'Ball Possession'])
                away_stats = self._calculate_team_stats(away_team, ['Fouls', 'Yellow Cards', 'Corner Kicks', 'Ball Possession'])
                
                # Calcular valores esperados
                self.predictions.at[idx, 'Expected Fouls'] = round((home_stats.get('Fouls Mean', 0) + away_stats.get('Fouls Mean', 0)) / 2, 1)
                self.predictions.at[idx, 'Expected Yellow Cards'] = round((home_stats.get('Yellow Cards Mean', 0) + away_stats.get('Yellow Cards Mean', 0)) / 2, 1)
                self.predictions.at[idx, 'Expected Corners'] = round((home_stats.get('Corner Kicks Mean', 0) + away_stats.get('Corner Kicks Mean', 0)) / 2, 1)
                self.predictions.at[idx, 'Expected Possession'] = round(home_stats.get('Ball Possession Mean', 50), 1)
            
            # Converter datas para datetime sem timezone
            if 'Game Date' in self.predictions.columns:
                self.predictions['Game Date'] = pd.to_datetime(self.predictions['Game Date']).dt.tz_localize(None)
            
            # Reordenar colunas
            columns = [
                'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win',
                'Game Date', 'Venue', 'Home Team Logo', 'Away Team Logo',
                'Expected Fouls', 'Expected Yellow Cards', 'Expected Corners', 'Expected Possession',
                'Home Team ID', 'Away Team ID', 'Fixture ID'
            ]
            self.predictions = self.predictions.reindex(columns=columns)
            
            logging.info("Predições realizadas com sucesso")
            logging.info(f"Formato da coluna Game Date: {self.predictions['Game Date'].dtype}")
            
        except Exception as e:
            logging.error(f"Erro ao realizar predições: {str(e)}")
            raise
    
    def plot_predictions(self) -> None:
        """Gera visualizações das predições"""
        try:
            if self.predictions is None or self.predictions.empty:
                logging.error("Não há predições para plotar")
                return
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            figures_dir = os.path.join(self.base_dir, 'figures')
            
            # Verificar se temos as colunas necessárias
            required_columns = ['Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win']
            if not all(col in self.predictions.columns for col in required_columns):
                logging.error(f"Colunas necessárias não encontradas. Colunas disponíveis: {self.predictions.columns.tolist()}")
                return
            
            # Criar rótulos para os jogos
            games = [f"{row['Home Team']} vs {row['Away Team']}" 
                    for _, row in self.predictions.iterrows()]
            
            # Preparar dados para o plot
            data = {
                'Home Win': self.predictions['Home Win'].values,
                'Draw': self.predictions['Draw'].values,
                'Away Win': self.predictions['Away Win'].values
            }
            
            # Criar figura com tamanho adequado
            plt.figure(figsize=(15, 8))
            
            # Criar DataFrame para plot
            df_plot = pd.DataFrame(data, index=games)
            
            # Definir cores para cada resultado
            colors = ['#2ecc71', '#95a5a6', '#e74c3c']  # Verde para vitória em casa, cinza para empate, vermelho para vitória fora
            
            # Criar o plot de barras empilhadas
            ax = df_plot.plot(
                kind='bar',
                stacked=True,
                color=colors,
                width=0.8
            )
            
            # Configurar o título e labels
            plt.title('Probabilidades de Resultado por Jogo', fontsize=14, pad=20)
            plt.xlabel('Jogos', fontsize=12)
            plt.ylabel('Probabilidade (%)', fontsize=12)
            
            # Rotacionar e ajustar labels do eixo x
            plt.xticks(rotation=45, ha='right')
            
            # Adicionar grid no eixo y
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adicionar valores nas barras
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f%%', label_type='center')
            
            # Adicionar legenda
            plt.legend(title='Resultado', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Salvar figura
            output_path = os.path.join(figures_dir, f'probabilities_{timestamp}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info(f"Visualização gerada com sucesso e salva em: {output_path}")
            
            # Imprimir informações sobre os dados plotados
            logging.info("\nResumo das probabilidades:")
            for idx, game in enumerate(games):
                home_win = data['Home Win'][idx]
                draw = data['Draw'][idx]
                away_win = data['Away Win'][idx]
                logging.info(f"{game}:")
                logging.info(f"  Vitória em Casa: {home_win:.1f}%")
                logging.info(f"  Empate: {draw:.1f}%")
                logging.info(f"  Vitória Fora: {away_win:.1f}%")
            
        except Exception as e:
            logging.error(f"Erro ao gerar visualizações: {str(e)}")
            logging.error(f"Shape dos dados: {self.predictions.shape if self.predictions is not None else 'None'}")
            raise
    
    def save_predictions(self) -> None:
        """Salva as predições geradas"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Criar caminhos completos para os arquivos
            results_dir = os.path.join(self.base_dir, 'results')
            csv_path = os.path.join(results_dir, f'predictions_{timestamp}.csv')
            pickle_path = os.path.join(results_dir, f'predictions_{timestamp}.pkl')
            
            # Salvar em formato CSV
            self.predictions.to_csv(csv_path, index=False)
            
            # Salvar em formato pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.predictions, f)
            
            # Salvar cópia com nome fixo para o webserver
            fixed_path = os.path.join(results_dir, 'predictions.pkl')
            with open(fixed_path, 'wb') as f:
                pickle.dump(self.predictions, f)
            
            logging.info(f"Predições salvas em {csv_path} e {pickle_path}")
            logging.info(f"Cópia fixa salva em {fixed_path}")
            
        except Exception as e:
            logging.error(f"Erro ao salvar predições: {str(e)}")
            raise
    
    def run_prediction_pipeline(self) -> None:
        """Executa o pipeline completo de predições"""
        try:
            logging.info("Iniciando pipeline de predições...")
            
            try:
                self.load_data()
                self.prepare_team_data()
                self.get_unplayed_games()
                self._prepare_features()
                self.make_predictions()
                self.plot_predictions()
                self.save_predictions()
                logging.info("Pipeline de predições concluído com sucesso")
            except ValueError as e:
                if str(e) in ["Não há jogos futuros para prever", "Fixtures precisam ser atualizados"]:
                    logging.warning("Pipeline encerrado: é necessário atualizar os fixtures")
                    logging.warning("Por favor, execute o script de coleta de dados para obter os próximos jogos")
                else:
                    raise
            
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