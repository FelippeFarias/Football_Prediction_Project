# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:13:02 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------


import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging


#------------------------------- DATA PROCESSING ------------------------------



def running_mean(data: Union[np.ndarray, pd.DataFrame],
                window_size: int = 5,
                team_col: Optional[str] = None,
                numeric_cols: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    Calcula a média móvel dos dados
    
    Args:
        data: Array ou DataFrame com os dados
        window_size: Tamanho da janela
        team_col: Coluna com identificador do time (opcional)
        numeric_cols: Lista de colunas numéricas (opcional)
    
    Returns:
        Array ou DataFrame com médias móveis
    """
    if isinstance(data, pd.DataFrame):
        if numeric_cols is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
        result = data.copy()
        
        if team_col is not None:
            # Calcular média móvel por time
            for col in numeric_cols:
                result[col] = result.groupby(team_col)[col].transform(
                    lambda x: x.rolling(window=window_size, min_periods=1).mean()
                )
        else:
            # Calcular média móvel geral
            for col in numeric_cols:
                result[col] = result[col].rolling(window=window_size, min_periods=1).mean()
                
        return result
    else:
        # Para arrays numpy
        data = np.asarray(data)
        if data.ndim == 1:
            # Expandir dimensões para 2D
            data = data.reshape(-1, 1)
            
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            # Usar convolve para média móvel eficiente
            weights = np.ones(window_size) / window_size
            result[:, i] = np.convolve(data[:, i], weights, mode='same')
            
            # Ajustar bordas
            for j in range(window_size - 1):
                if j < window_size // 2:
                    # Início
                    result[j, i] = np.mean(data[:(j + window_size//2 + 1), i])
                else:
                    # Fim
                    result[-(window_size-j), i] = np.mean(data[-(window_size-j):, i])
                    
        return result.squeeze() if data.shape[1] == 1 else result



def calculate_momentum(df: pd.DataFrame,
                      result_col: str = 'Result Indicator',
                      team_id_col: str = 'Team ID',
                      window: int = 5) -> pd.DataFrame:
    """
    Calcula momentum baseado em resultados recentes
    
    Args:
        df: DataFrame com dados
        result_col: Nome da coluna de resultado
        team_id_col: Nome da coluna de ID do time
        window: Tamanho da janela
    
    Returns:
        DataFrame com momentum calculado
    """
    df_momentum = df.copy()
    
    # Mapear resultados para pontos
    points_map = {0: 0, 1: 1, 2: 3}  # 0=derrota, 1=empate, 2=vitória
    df_momentum['points'] = df_momentum[result_col].map(points_map)
    
    # Calcular momentum por time
    df_momentum['momentum'] = df_momentum.groupby(team_id_col)['points'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # Normalizar momentum entre 0 e 1
    momentum_max = df_momentum['momentum'].max()
    momentum_min = df_momentum['momentum'].min()
    
    if momentum_max > momentum_min:
        df_momentum['momentum'] = (df_momentum['momentum'] - momentum_min) / (momentum_max - momentum_min)
    else:
        df_momentum['momentum'] = 0
        
    # Remover coluna temporária
    df_momentum.drop('points', axis=1, inplace=True)
    
    return df_momentum



def calculate_fatigue(df: pd.DataFrame,
                     date_col: str = 'Game Date',
                     team_id_col: str = 'Team ID',
                     window: int = 30) -> pd.DataFrame:
    """
    Calcula fadiga baseada em jogos recentes
    
    Args:
        df: DataFrame com dados
        date_col: Nome da coluna de data
        team_id_col: Nome da coluna de ID do time
        window: Janela em dias para considerar
        
    Returns:
        DataFrame com fadiga calculada
    """
    df_fatigue = df.copy()
    
    # Converter datas
    df_fatigue[date_col] = pd.to_datetime(df_fatigue[date_col])
    
    # Ordenar por time e data
    df_fatigue.sort_values([team_id_col, date_col], inplace=True)
    
    # Calcular dias desde último jogo por time
    df_fatigue['days_since_last'] = df_fatigue.groupby(team_id_col)[date_col].diff().dt.days
    
    # Preencher primeiro jogo de cada time com valor alto
    df_fatigue['days_since_last'].fillna(window, inplace=True)
    
    # Calcular fadiga (inverso dos dias de descanso)
    df_fatigue['fatigue'] = 1 - (df_fatigue['days_since_last'] / window)
    
    # Limitar entre 0 e 1
    df_fatigue['fatigue'] = df_fatigue['fatigue'].clip(0, 1)
    
    # Remover coluna temporária
    df_fatigue.drop('days_since_last', axis=1, inplace=True)
    
    return df_fatigue



def average_stats_df(window_size: int,
                    team_list: List[str],
                    team_fixture_id_dict: Dict[str, List[str]],
                    game_stats: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Calcula estatísticas médias dos times
    
    Args:
        window_size: Tamanho da janela móvel
        team_list: Lista de times
        team_fixture_id_dict: Dicionário com IDs dos jogos por time
        game_stats: Dicionário com estatísticas dos jogos
    
    Returns:
        DataFrame com estatísticas médias
    """
    
    all_stats_dict_list = []
    
    for team in team_list:
        fixture_id_list = team_fixture_id_dict[team]
        
        for fix_id in fixture_id_list:
            if fix_id not in game_stats[team]:
                continue
                
            game_stats_dict = {}
            game_stats_dict['Team'] = team
            game_stats_dict['Fixture ID'] = fix_id
            
            # Obter estatísticas do jogo atual
            current_game_stats = game_stats[team][fix_id]
            
            # Adicionar Team ID
            team_idx = 0 if current_game_stats['Team Identifier'].iloc[0] == 1 else 1
            game_stats_dict['Team ID'] = current_game_stats['Team ID'].iloc[team_idx]
            
            # Adicionar Result Indicator
            if 'Goals' in current_game_stats.columns:
                goals = float(current_game_stats['Goals'].iloc[team_idx])
                opponent_goals = float(current_game_stats['Goals'].iloc[1 - team_idx])
                
                if goals > opponent_goals:
                    game_stats_dict['Result Indicator'] = 2  # Vitória
                elif goals == opponent_goals:
                    game_stats_dict['Result Indicator'] = 1  # Empate
                else:
                    game_stats_dict['Result Indicator'] = 0  # Derrota
            
            # Encontrar índice do jogo atual
            current_game_index = fixture_id_list.index(fix_id)
            
            # Calcular índices para a janela móvel
            start_index = max(0, current_game_index - window_size)
            
            # Coletar estatísticas da janela
            window_stats = {}
            for i in range(start_index, current_game_index):
                if fixture_id_list[i] in game_stats[team]:
                    game_data = game_stats[team][fixture_id_list[i]]
                    team_idx = 0 if game_data['Team Identifier'].iloc[0] == 1 else 1
                    for col in game_data.columns:
                        if col not in ['Team Identifier', 'Team ID', 'Team', 'Game Date']:
                            if col not in window_stats:
                                window_stats[col] = []
                            try:
                                # Converter para float, ignorando valores não numéricos
                                value = float(game_data[col].iloc[team_idx])
                                window_stats[col].append(value)
                            except (ValueError, TypeError):
                                continue
            
            # Calcular médias e diferenças
            for col in current_game_stats.columns:
                if col not in ['Team Identifier', 'Team ID', 'Team', 'Game Date']:
                    try:
                        current_value = float(current_game_stats[col].iloc[team_idx])
                        if col in window_stats and window_stats[col]:
                            mean_value = np.mean(window_stats[col])
                            game_stats_dict[f'{col} Mean'] = mean_value
                            game_stats_dict[f'{col} Diff'] = current_value - mean_value
                        else:
                            game_stats_dict[f'{col} Mean'] = current_value
                            game_stats_dict[f'{col} Diff'] = 0
                    except (ValueError, TypeError):
                        continue
            
            # Adicionar data do jogo
            game_stats_dict['Game Date'] = current_game_stats['Game Date'].iloc[0]
            
            all_stats_dict_list.append(game_stats_dict)
    
    # Criar DataFrame
    if not all_stats_dict_list:
        return pd.DataFrame()
        
    df_output = pd.DataFrame(all_stats_dict_list)
    
    return df_output



def mod_df(df: pd.DataFrame,
           window_size: int = 5) -> pd.DataFrame:
    """
    Modifica DataFrame adicionando features
    
    Args:
        df: DataFrame original
        window_size: Tamanho da janela
        
    Returns:
        DataFrame modificado
    """
    df_mod = df.copy()
    
    # Calcular médias móveis apenas para colunas que existem
    numeric_cols = []
    for col in ['Goals Mean', 'Shots on Goal Mean']:
        if col in df_mod.columns:
            numeric_cols.append(col)
    
    if numeric_cols:
        df_mod = running_mean(df_mod, window_size, 'Team ID', numeric_cols)
    
    # Adicionar momentum se Result Indicator existir
    if 'Result Indicator' in df_mod.columns:
        df_mod = calculate_momentum(df_mod, result_col='Result Indicator', team_id_col='Team ID', window=window_size)
    
    # Adicionar fadiga se as colunas necessárias existirem
    if 'Game Date' in df_mod.columns and 'Team ID' in df_mod.columns:
        df_mod = calculate_fatigue(df_mod, team_id_col='Team ID')
    
    return df_mod



def combining_fixture_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina características dos times em um único DataFrame por fixture
    
    Args:
        df: DataFrame com estatísticas dos times
        
    Returns:
        DataFrame combinado
    """
    df_output = pd.DataFrame()
    
    # Adicionar colunas relevantes
    for col in df.columns:
        if col.endswith('Mean') or col.endswith('Diff') or col in ['Team ID', 'Fixture ID', 'Result Indicator', 'Game Date']:
            df_output[col] = df[col]
    
    return df_output



def creating_ml_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria DataFrame para machine learning
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame processado para ML
    """
    # Modificar DataFrame
    modified_df = mod_df(df)
    
    # Combinar características
    df_output = combining_fixture_id(modified_df)
    
    return df_output




