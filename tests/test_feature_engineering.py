import pytest
import numpy as np
import pandas as pd
from ml_functions.feature_engineering_functions import (
    running_mean,
    calculate_momentum,
    calculate_fatigue,
    average_stats_df,
    mod_df
)

@pytest.fixture
def sample_data():
    """Fixture para criar dados de exemplo para testes"""
    np.random.seed(42)
    return np.random.rand(10)

@pytest.fixture
def sample_df():
    """Fixture para criar DataFrame de exemplo"""
    data = {
        'Team ID': ['Team1', 'Team2'] * 5,
        'Goals': [2, 1, 3, 0, 1, 2, 0, 1, 2, 3],
        'Shots': [10, 8, 12, 6, 9, 11, 7, 8, 10, 13],
        'Game Date': pd.date_range(start='2024-01-01', periods=10)
    }
    return pd.DataFrame(data)

@pytest.mark.feature_engineering
def test_running_mean():
    """Testa a função de média móvel"""
    data = np.array([1, 2, 3, 4, 5])
    window = 3
    expected = np.array([np.nan, np.nan, 2, 3, 4])
    result = running_mean(data, window)
    
    # Verificar valores não-NA
    mask = ~np.isnan(result)
    np.testing.assert_array_almost_equal(result[mask], expected[mask])
    
    # Verificar número correto de valores NA
    assert np.sum(np.isnan(result)) == window - 1

@pytest.mark.feature_engineering
def test_calculate_momentum(sample_df):
    """Testa o cálculo do momentum"""
    sample_df['result'] = [1, 0, 2, 1, 1, 0, 2, 1, 0, 2]
    momentum_df = calculate_momentum(sample_df, result_col='result', window=3)
    
    assert len(momentum_df) == len(sample_df)
    assert 'momentum' in momentum_df.columns
    assert all(0 <= m <= 1 for m in momentum_df['momentum'])

@pytest.mark.feature_engineering
def test_calculate_fatigue(sample_df):
    """Testa o cálculo da fadiga"""
    fatigue_df = calculate_fatigue(sample_df, date_col='Game Date', team_col='Team ID', window=30)
    
    assert len(fatigue_df) == len(sample_df)
    assert 'fatigue' in fatigue_df.columns
    assert all(0 <= f <= 1 for f in fatigue_df['fatigue'])

@pytest.mark.feature_engineering
def test_average_stats_df():
    """Testa o cálculo de estatísticas médias"""
    team_list = ['Team1', 'Team2']
    team_fixture_id_dict = {
        'Team1': ['0', '2', '4'],
        'Team2': ['1', '3', '5']
    }
    game_stats = {
        'Team1': {
            '0': sample_df.iloc[[0]],
            '2': sample_df.iloc[[2]],
            '4': sample_df.iloc[[4]]
        },
        'Team2': {
            '1': sample_df.iloc[[1]],
            '3': sample_df.iloc[[3]],
            '5': sample_df.iloc[[5]]
        }
    }
    
    result = average_stats_df(2, team_list, team_fixture_id_dict, game_stats)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'Team Av Goals Diff' in result.columns
    assert 'Team Av Shots Diff' in result.columns

@pytest.mark.feature_engineering
def test_mod_df(sample_df):
    """Testa as modificações no DataFrame"""
    result = mod_df(sample_df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) == len(sample_df)
    # Verificar se novas colunas foram adicionadas
    assert any(col.startswith('Team Av') for col in result.columns) 