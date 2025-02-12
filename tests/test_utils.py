import pytest
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ml_functions.utils import (
    load_latest_data,
    save_model,
    load_model,
    format_predictions,
    calculate_stake,
    get_betting_advice
)
from ml_functions.prediction_functions import prepare_prediction_data

@pytest.fixture
def test_dir(tmpdir):
    """Fixture para criar diretório temporário para testes"""
    test_dir = tmpdir.mkdir('test_utils')
    return str(test_dir)

@pytest.fixture
def sample_data():
    """Fixture para criar dados de exemplo"""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    return df, X, y, feature_cols

@pytest.fixture
def sample_model(sample_data):
    """Fixture para criar modelo de exemplo"""
    _, X, y, _ = sample_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

@pytest.mark.utils
def test_load_latest_data(test_dir, sample_data):
    """Testa o carregamento dos dados mais recentes"""
    df, _, _, _ = sample_data
    
    # Criar múltiplos arquivos com timestamps diferentes
    for i in range(3):
        timestamp = datetime.now().strftime(f'%Y%m%d_%H%M%S_{i}')
        filename = f'data_{timestamp}.csv'
        filepath = os.path.join(test_dir, filename)
        df.to_csv(filepath, index=False)
    
    # Carregar dados mais recentes
    loaded_df, latest_file = load_latest_data(
        data_dir=test_dir,
        pattern='data_*.csv'
    )
    
    assert isinstance(loaded_df, pd.DataFrame)
    assert os.path.exists(os.path.join(test_dir, latest_file))
    pd.testing.assert_frame_equal(loaded_df, df)

@pytest.mark.utils
def test_save_and_load_model(test_dir, sample_model, sample_data):
    """Testa o salvamento e carregamento do modelo"""
    _, X, y, _ = sample_data
    
    # Criar transformadores
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Salvar modelo e transformadores
    save_model(
        model=sample_model,
        scaler=scaler,
        label_encoder=le,
        save_dir=test_dir,
        window_size=10
    )
    
    # Carregar modelo e transformadores
    loaded_model, loaded_scaler, loaded_le = load_model(
        model_dir=test_dir,
        window_size=10
    )
    
    # Verificar se os objetos foram carregados corretamente
    assert isinstance(loaded_model, RandomForestClassifier)
    assert isinstance(loaded_scaler, StandardScaler)
    assert isinstance(loaded_le, LabelEncoder)
    
    # Verificar predições
    original_pred = sample_model.predict(X)
    loaded_pred = loaded_model.predict(X)
    np.testing.assert_array_equal(original_pred, loaded_pred)

@pytest.mark.utils
def test_prepare_prediction_data(sample_data):
    """Testa a preparação dos dados para predição"""
    df, _, _, feature_cols = sample_data
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    
    prepared_data = prepare_prediction_data(
        df=df,
        scaler=scaler,
        feature_cols=feature_cols
    )
    
    assert isinstance(prepared_data, np.ndarray)
    assert prepared_data.shape[1] == len(feature_cols)
    assert np.allclose(prepared_data.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(prepared_data.std(axis=0), 1, atol=1e-10)

@pytest.mark.utils
def test_format_predictions(sample_data, sample_model):
    """Testa a formatação das predições"""
    df, X, y, _ = sample_data
    
    # Fazer predições
    y_pred = sample_model.predict(X)
    y_prob = sample_model.predict_proba(X)
    
    # Criar LabelEncoder
    le = LabelEncoder()
    le.fit(y)
    
    formatted_df = format_predictions(
        y_pred=y_pred,
        y_prob=y_prob,
        label_encoder=le,
        df=df
    )
    
    assert isinstance(formatted_df, pd.DataFrame)
    assert 'prediction' in formatted_df.columns
    assert all(col in formatted_df.columns for col in ['prob_0', 'prob_1', 'prob_2'])
    assert len(formatted_df) == len(df)

@pytest.mark.utils
def test_calculate_stake():
    """Testa o cálculo do stake"""
    # Caso favorável
    stake = calculate_stake(
        prob=0.7,
        odds=2.0,
        bankroll=1000,
        max_stake=0.1
    )
    assert isinstance(stake, float)
    assert 0 <= stake <= 100
    
    # Caso desfavorável
    stake = calculate_stake(
        prob=0.3,
        odds=2.0,
        bankroll=1000,
        max_stake=0.1
    )
    assert stake == 0

@pytest.mark.utils
def test_get_betting_advice():
    """Testa as recomendações de apostas"""
    # Criar DataFrame de predições
    pred_df = pd.DataFrame({
        'home_team': ['TeamA', 'TeamB', 'TeamC'],
        'away_team': ['TeamX', 'TeamY', 'TeamZ'],
        'prob_home_win': [0.7, 0.4, 0.5],
        'prob_draw': [0.2, 0.3, 0.3],
        'prob_away_win': [0.1, 0.3, 0.2],
        'odds_home': [1.5, 2.0, 1.8],
        'odds_draw': [3.5, 3.0, 3.2],
        'odds_away': [6.0, 2.5, 4.0]
    })
    
    advice = get_betting_advice(
        pred_df=pred_df,
        threshold=0.6,
        bankroll=1000
    )
    
    assert isinstance(advice, list)
    for bet in advice:
        assert isinstance(bet, dict)
        assert all(key in bet for key in ['match', 'bet_type', 'probability', 'odds', 'stake'])
        assert 0 <= bet['probability'] <= 1
        assert bet['odds'] > 1
        assert 0 <= bet['stake'] <= 100

@pytest.mark.utils
def test_edge_cases():
    """Testa casos extremos das funções utilitárias"""
    # Teste com probabilidade 1.0
    stake_max = calculate_stake(
        prob=1.0,
        odds=2.0,
        bankroll=1000,
        max_stake=0.1
    )
    assert stake_max == 100  # stake máximo
    
    # Teste com probabilidade 0.0
    stake_min = calculate_stake(
        prob=0.0,
        odds=2.0,
        bankroll=1000,
        max_stake=0.1
    )
    assert stake_min == 0  # sem aposta
    
    # Teste com odds muito baixas
    stake_low_odds = calculate_stake(
        prob=0.9,
        odds=1.1,
        bankroll=1000,
        max_stake=0.1
    )
    assert stake_low_odds == 0  # sem aposta devido a odds desfavoráveis 