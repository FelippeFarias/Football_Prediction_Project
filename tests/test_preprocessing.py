import pytest
import numpy as np
import pandas as pd
from ml_functions.preprocessing_functions import (
    prepare_features,
    handle_missing_values,
    remove_outliers,
    encode_categorical_features,
    create_time_features,
    create_interaction_features
)

@pytest.fixture
def sample_df():
    """Fixture para criar DataFrame de exemplo com valores faltantes e outliers"""
    np.random.seed(42)
    data = {
        'numeric_1': [1, 2, np.nan, 4, 100],  # outlier em 100
        'numeric_2': [10, np.nan, 30, 40, 50],
        'category_1': ['A', 'B', None, 'A', 'C'],
        'category_2': ['X', 'Y', 'Z', None, 'X'],
        'date': pd.date_range(start='2024-01-01', periods=5)
    }
    return pd.DataFrame(data)

@pytest.mark.preprocessing
def test_prepare_features(sample_df):
    """Testa a preparação de features"""
    feature_cols = ['numeric_1', 'numeric_2']
    target_col = 'category_1'
    
    X_train, X_test, y_train, y_test, scaler, le = prepare_features(
        df=sample_df.fillna(0),  # Preencher NA para este teste
        feature_cols=feature_cols,
        target_col=target_col
    )
    
    # Verificar dimensões
    assert X_train.shape[1] == len(feature_cols)
    assert len(y_train) == X_train.shape[0]
    
    # Verificar se os dados foram normalizados
    assert np.all(np.abs(X_train.mean(axis=0)) < 1e-10)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-10)

@pytest.mark.preprocessing
def test_handle_missing_values(sample_df):
    """Testa o tratamento de valores faltantes"""
    result = handle_missing_values(
        sample_df,
        numeric_strategy='mean',
        categorical_strategy='mode'
    )
    
    # Verificar se não há mais valores faltantes
    assert not result.isnull().any().any()
    
    # Verificar se os valores foram preenchidos corretamente
    numeric_cols = ['numeric_1', 'numeric_2']
    for col in numeric_cols:
        assert not pd.isna(result[col]).any()
        
    categorical_cols = ['category_1', 'category_2']
    for col in categorical_cols:
        assert not pd.isna(result[col]).any()

@pytest.mark.preprocessing
def test_remove_outliers(sample_df):
    """Testa a remoção de outliers"""
    numeric_cols = ['numeric_1', 'numeric_2']
    result = remove_outliers(
        sample_df,
        cols=numeric_cols,
        n_std=3.0
    )
    
    # Verificar se o outlier foi removido
    assert len(result) < len(sample_df)
    assert 100 not in result['numeric_1'].values

@pytest.mark.preprocessing
def test_encode_categorical_features(sample_df):
    """Testa a codificação de features categóricas"""
    cat_cols = ['category_1', 'category_2']
    
    # Testar Label Encoding
    result_label, encoders = encode_categorical_features(
        sample_df.fillna('missing'),
        cat_cols=cat_cols,
        encoding='label'
    )
    
    assert all(result_label[col].dtype == 'int64' for col in cat_cols)
    assert len(encoders) == len(cat_cols)
    
    # Testar One-Hot Encoding
    result_onehot, _ = encode_categorical_features(
        sample_df.fillna('missing'),
        cat_cols=cat_cols,
        encoding='onehot'
    )
    
    assert all(col not in result_onehot.columns for col in cat_cols)
    assert len(result_onehot.columns) > len(sample_df.columns)

@pytest.mark.preprocessing
def test_create_time_features(sample_df):
    """Testa a criação de features temporais"""
    result = create_time_features(sample_df, 'date')
    
    expected_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']
    assert all(feat in result.columns for feat in expected_features)
    assert 'date' not in result.columns  # verificar se a coluna original foi removida
    
    # Verificar tipos de dados
    assert result['year'].dtype == 'int64'
    assert result['month'].dtype == 'int64'
    assert result['is_weekend'].dtype == 'int64'

@pytest.mark.preprocessing
def test_create_interaction_features(sample_df):
    """Testa a criação de features de interação"""
    feature_pairs = [('numeric_1', 'numeric_2')]
    result = create_interaction_features(
        sample_df.fillna(0),  # Preencher NA para este teste
        feature_pairs=feature_pairs
    )
    
    expected_feature = 'numeric_1_numeric_2_interaction'
    assert expected_feature in result.columns
    
    # Verificar se a interação foi calculada corretamente
    expected_value = sample_df['numeric_1'].fillna(0) * sample_df['numeric_2'].fillna(0)
    pd.testing.assert_series_equal(
        result[expected_feature],
        expected_value,
        check_names=False
    ) 