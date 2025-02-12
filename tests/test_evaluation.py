import pytest
import numpy as np
import pandas as pd
from ml_functions.evaluation_functions import (
    evaluate_classification_metrics,
    evaluate_betting_metrics,
    evaluate_calibration,
    evaluate_feature_importance,
    evaluate_predictions_over_time
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

@pytest.fixture
def sample_predictions_df():
    """Fixture para criar DataFrame de predições"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'game_id': range(n_samples),
        'date': pd.date_range(start='2024-01-01', periods=n_samples),
        'prediction': np.random.randint(0, 3, n_samples),
        'probability': np.random.rand(n_samples),
        'odds': np.random.uniform(1.5, 4.0, n_samples),
        'result': np.random.randint(0, 3, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.mark.evaluation
def test_evaluate_classification_metrics(sample_predictions):
    """Testa o cálculo de métricas de classificação"""
    y_true, y_pred, y_prob = sample_predictions
    
    metrics = evaluate_classification_metrics(y_true, y_pred, y_prob)
    
    assert isinstance(metrics, dict)
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
    if y_prob is not None:
        expected_metrics.extend(['roc_auc', 'log_loss', 'brier_score'])
    
    assert all(metric in metrics for metric in expected_metrics)
    assert all(0 <= metrics[m] <= 1 for m in ['accuracy', 'precision', 'recall', 'f1'])
    
    if y_prob is not None:
        assert 0 <= metrics['roc_auc'] <= 1
        assert metrics['log_loss'] >= 0
        assert 0 <= metrics['brier_score'] <= 1

@pytest.mark.evaluation
def test_evaluate_betting_metrics(sample_predictions):
    """Testa o cálculo de métricas para apostas"""
    y_true, y_pred, y_prob = sample_predictions
    
    odds = {
        'Derrota': 2.5,
        'Empate': 3.5,
        'Vitória': 2.5
    }
    
    metrics = evaluate_betting_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        odds=odds,
        threshold=0.5
    )
    
    assert isinstance(metrics, dict)
    for result_type in ['Derrota', 'Empate', 'Vitória']:
        if result_type in metrics:
            result_metrics = metrics[result_type]
            assert 'precisão' in result_metrics
            assert 'roi_esperado' in result_metrics
            assert 'stake_sugerido' in result_metrics
            assert 'n_apostas' in result_metrics
            
            assert 0 <= result_metrics['precisão'] <= 100
            assert result_metrics['stake_sugerido'] >= 0
            assert result_metrics['stake_sugerido'] <= 10
            assert isinstance(result_metrics['n_apostas'], int)

@pytest.mark.evaluation
def test_evaluate_calibration(sample_predictions):
    """Testa a avaliação de calibração"""
    y_true, _, y_prob = sample_predictions
    
    prob_true, prob_pred, counts = evaluate_calibration(
        y_true=y_true,
        y_prob=y_prob[:, 1],  # usar apenas uma classe para teste
        n_bins=5
    )
    
    assert isinstance(prob_true, np.ndarray)
    assert isinstance(prob_pred, np.ndarray)
    assert isinstance(counts, np.ndarray)
    
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) == len(counts)
    
    assert np.all((prob_true >= 0) & (prob_true <= 1))
    assert np.all((prob_pred >= 0) & (prob_pred <= 1))
    assert np.all(counts > 0)

@pytest.mark.evaluation
def test_evaluate_feature_importance(sample_predictions):
    """Testa a avaliação de importância das features"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Criar dados de exemplo
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Treinar modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    importances = evaluate_feature_importance(
        model=model,
        feature_names=feature_names,
        threshold=0.1
    )
    
    assert isinstance(importances, dict)
    assert all(isinstance(v, float) for v in importances.values())
    assert all(f in feature_names for f in importances.keys())
    assert all(v >= 0.1 for v in importances.values())
    assert sum(importances.values()) <= 1.0

@pytest.mark.evaluation
def test_evaluate_predictions_over_time(sample_predictions_df):
    """Testa a avaliação de predições ao longo do tempo"""
    results = evaluate_predictions_over_time(
        predictions=sample_predictions_df,
        actual_results=sample_predictions_df,  # usar mesmo DataFrame para teste
        window_size=10
    )
    
    assert isinstance(results, pd.DataFrame)
    assert 'date' in results.columns
    assert 'accuracy' in results.columns
    assert 'roi' in results.columns
    
    assert len(results) == len(sample_predictions_df) - 10  # devido à janela móvel
    assert all(isinstance(d, pd.Timestamp) for d in results['date'])
    assert all(0 <= acc <= 1 for acc in results['accuracy']) 