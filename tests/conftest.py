import pytest
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

@pytest.fixture(scope='session')
def test_dir(tmpdir_factory):
    """Fixture para criar diretório temporário para testes"""
    test_dir = tmpdir_factory.mktemp('test_data')
    return str(test_dir)

@pytest.fixture(scope='session')
def sample_data():
    """Fixture para criar dados de exemplo"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Criar features
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    # Criar DataFrame
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    return df, X, y, feature_cols

@pytest.fixture(scope='session')
def sample_predictions():
    """Fixture para criar predições de exemplo"""
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = np.random.randint(0, 3, n_samples)
    y_prob = np.random.rand(n_samples, 3)
    y_prob = y_prob / y_prob.sum(axis=1)[:, np.newaxis]  # normalizar probabilidades
    
    return y_true, y_pred, y_prob

@pytest.fixture(scope='session')
def sample_model(sample_data):
    """Fixture para criar modelo de exemplo"""
    _, X, y, _ = sample_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture(scope='session')
def sample_transformers(sample_data):
    """Fixture para criar transformadores"""
    _, X, y, _ = sample_data
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return scaler, le

@pytest.fixture(scope='session')
def sample_predictions_df():
    """Fixture para criar DataFrame de predições"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'game_id': range(n_samples),
        'date': pd.date_range(start='2024-01-01', periods=n_samples),
        'home_team': [f'Team{i}' for i in range(n_samples)],
        'away_team': [f'Opponent{i}' for i in range(n_samples)],
        'prediction': np.random.randint(0, 3, n_samples),
        'probability': np.random.rand(n_samples),
        'odds': np.random.uniform(1.5, 4.0, n_samples),
        'result': np.random.randint(0, 3, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture(scope='session')
def sample_logger(test_dir):
    """Fixture para criar logger de teste"""
    import logging
    
    log_file = os.path.join(test_dir, 'test.log')
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    
    # Adicionar handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Configurar formato
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Adicionar handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def pytest_configure(config):
    """Configuração do pytest"""
    config.addinivalue_line(
        "markers",
        "feature_engineering: mark test as feature engineering test"
    )
    config.addinivalue_line(
        "markers",
        "preprocessing: mark test as preprocessing test"
    )
    config.addinivalue_line(
        "markers",
        "modeling: mark test as modeling test"
    )
    config.addinivalue_line(
        "markers",
        "evaluation: mark test as evaluation test"
    )
    config.addinivalue_line(
        "markers",
        "prediction: mark test as prediction test"
    )
    config.addinivalue_line(
        "markers",
        "visualization: mark test as visualization test"
    )
    config.addinivalue_line(
        "markers",
        "metrics: mark test as metrics test"
    )
    config.addinivalue_line(
        "markers",
        "logging: mark test as logging test"
    )
    config.addinivalue_line(
        "markers",
        "utils: mark test as utils test"
    ) 