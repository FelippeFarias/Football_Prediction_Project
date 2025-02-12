import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from ml_functions.modeling_functions import (
    optimize_model,
    train_model,
    predict_proba_calibrated,
    ensemble_predictions,
    get_model_explanation
)

@pytest.fixture
def sample_data():
    """Fixture para criar dados de exemplo"""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)  # 3 classes
    return X, y

@pytest.fixture
def base_model():
    """Fixture para criar modelo base"""
    return RandomForestClassifier(random_state=42)

@pytest.mark.modeling
def test_optimize_model(sample_data, base_model):
    """Testa a otimização de hiperparâmetros"""
    X, y = sample_data
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5]
    }
    
    model, best_params = optimize_model(
        model=base_model,
        param_grid=param_grid,
        X_train=X,
        y_train=y,
        cv=2
    )
    
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params
    assert 'max_depth' in best_params

@pytest.mark.modeling
def test_train_model(sample_data, base_model):
    """Testa o treinamento do modelo"""
    X, y = sample_data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    model, metrics = train_model(
        model=base_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Verificar se as métricas estão no intervalo correto
    assert all(0 <= v <= 1 for v in metrics.values())

@pytest.mark.modeling
def test_predict_proba_calibrated(sample_data, base_model):
    """Testa a calibração de probabilidades"""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train = y[:80]
    
    # Treinar modelo
    base_model.fit(X_train, y_train)
    
    # Calibrar probabilidades
    calibrated_probs = predict_proba_calibrated(
        model=base_model,
        X=X_test,
        calibration_method='isotonic'
    )
    
    assert isinstance(calibrated_probs, np.ndarray)
    assert calibrated_probs.shape[0] == X_test.shape[0]
    assert calibrated_probs.shape[1] == len(np.unique(y))
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0)
    assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))

@pytest.mark.modeling
def test_ensemble_predictions(sample_data):
    """Testa as predições do ensemble"""
    X, y = sample_data
    X_train, X_test = X[:80], X[80:]
    y_train = y[:80]
    
    # Criar múltiplos modelos
    models = [
        RandomForestClassifier(n_estimators=10, random_state=i).fit(X_train, y_train)
        for i in range(3)
    ]
    
    # Fazer predições com ensemble
    y_pred, y_prob = ensemble_predictions(
        models=models,
        X=X_test,
        weights=[0.4, 0.3, 0.3]
    )
    
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_prob, np.ndarray)
    assert len(y_pred) == X_test.shape[0]
    assert y_prob.shape == (X_test.shape[0], len(np.unique(y)))
    assert np.allclose(y_prob.sum(axis=1), 1.0)

@pytest.mark.modeling
def test_get_model_explanation(sample_data, base_model):
    """Testa a explicação do modelo"""
    X, y = sample_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Treinar modelo
    base_model.fit(X, y)
    
    explanation = get_model_explanation(
        model=base_model,
        X=X,
        feature_names=feature_names,
        n_features=3
    )
    
    assert isinstance(explanation, dict)
    assert 'feature_importance' in explanation
    assert len(explanation['feature_importance']) <= 3
    assert all(isinstance(v, float) for v in explanation['feature_importance'].values())
    assert all(f in feature_names for f in explanation['feature_importance'].keys()) 