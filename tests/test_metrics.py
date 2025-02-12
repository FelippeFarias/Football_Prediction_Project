import pytest
import numpy as np
from ml_functions.metrics_functions import (
    calculate_betting_metrics,
    calculate_model_metrics,
    calculate_calibration_metrics
)

@pytest.fixture
def sample_predictions():
    """Fixture para criar predições de exemplo"""
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = np.random.randint(0, 3, n_samples)
    y_prob = np.random.rand(n_samples, 3)
    y_prob = y_prob / y_prob.sum(axis=1)[:, np.newaxis]  # normalizar probabilidades
    
    return y_true, y_pred, y_prob

@pytest.mark.metrics
def test_calculate_betting_metrics(sample_predictions):
    """Testa o cálculo de métricas para apostas"""
    y_true, y_pred, y_prob = sample_predictions
    
    metrics = calculate_betting_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=0.5
    )
    
    assert isinstance(metrics, dict)
    for result_type in metrics:
        result_metrics = metrics[result_type]
        assert 'precisão' in result_metrics
        assert 'roi_esperado' in result_metrics
        assert 'stake_sugerido' in result_metrics
        assert 'n_apostas' in result_metrics
        
        # Verificar intervalos válidos
        assert 0 <= result_metrics['precisão'] <= 100
        assert result_metrics['stake_sugerido'] >= 0
        assert result_metrics['stake_sugerido'] <= 10
        assert isinstance(result_metrics['n_apostas'], int)
        assert result_metrics['n_apostas'] >= 0

@pytest.mark.metrics
def test_calculate_model_metrics(sample_predictions):
    """Testa o cálculo de métricas do modelo"""
    y_true, y_pred, y_prob = sample_predictions
    
    metrics = calculate_model_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )
    
    assert isinstance(metrics, dict)
    expected_metrics = ['acurácia', 'precisão', 'recall', 'f1', 'log_loss', 'brier_score']
    
    # Verificar presença de todas as métricas
    assert all(metric in metrics for metric in expected_metrics)
    
    # Verificar intervalos válidos
    percentage_metrics = ['acurácia', 'precisão', 'recall', 'f1']
    assert all(0 <= metrics[m] <= 100 for m in percentage_metrics)
    
    # Verificar métricas de perda
    assert metrics['log_loss'] >= 0
    assert 0 <= metrics['brier_score'] <= 1

@pytest.mark.metrics
def test_calculate_calibration_metrics(sample_predictions):
    """Testa o cálculo de métricas de calibração"""
    y_true, _, y_prob = sample_predictions
    
    prob_true, prob_pred, counts = calculate_calibration_metrics(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=5
    )
    
    # Verificar tipos de retorno
    assert isinstance(prob_true, np.ndarray)
    assert isinstance(prob_pred, np.ndarray)
    assert isinstance(counts, np.ndarray)
    
    # Verificar dimensões
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) == len(counts)
    
    # Verificar intervalos válidos
    assert np.all((prob_true >= 0) & (prob_true <= 1))
    assert np.all((prob_pred >= 0) & (prob_pred <= 1))
    assert np.all(counts > 0)

@pytest.mark.metrics
def test_betting_metrics_edge_cases():
    """Testa casos extremos das métricas de apostas"""
    # Caso perfeito
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_prob = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    metrics = calculate_betting_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=0.5
    )
    
    for result_type in metrics:
        assert metrics[result_type]['precisão'] == 100
        assert metrics[result_type]['roi_esperado'] > 0
    
    # Caso completamente errado
    y_pred = np.array([2, 0, 1])
    metrics = calculate_betting_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=0.5
    )
    
    for result_type in metrics:
        assert metrics[result_type]['precisão'] == 0
        assert metrics[result_type]['roi_esperado'] < 0

@pytest.mark.metrics
def test_model_metrics_edge_cases():
    """Testa casos extremos das métricas do modelo"""
    # Caso perfeito
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_prob = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    metrics = calculate_model_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )
    
    assert metrics['acurácia'] == 100
    assert metrics['precisão'] == 100
    assert metrics['recall'] == 100
    assert metrics['f1'] == 100
    assert metrics['brier_score'] == 0
    
    # Caso completamente errado
    y_pred = np.array([2, 0, 1])
    y_prob = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    metrics = calculate_model_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )
    
    assert metrics['acurácia'] == 0
    assert metrics['precisão'] == 0
    assert metrics['recall'] == 0
    assert metrics['f1'] == 0
    assert metrics['brier_score'] > 0 