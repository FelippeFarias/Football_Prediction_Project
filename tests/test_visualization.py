import pytest
import numpy as np
import pandas as pd
import os
from ml_functions.visualization_functions import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_prediction_distribution,
    plot_model_comparison
)

@pytest.fixture
def sample_data():
    """Fixture para criar dados de exemplo"""
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = np.random.randint(0, 3, n_samples)
    y_prob = np.random.rand(n_samples, 3)
    y_prob = y_prob / y_prob.sum(axis=1)[:, np.newaxis]  # normalizar probabilidades
    
    return y_true, y_pred, y_prob

@pytest.fixture
def test_dir(tmpdir):
    """Fixture para criar diretório temporário para testes"""
    test_dir = tmpdir.mkdir('test_plots')
    return str(test_dir)

@pytest.mark.visualization
def test_plot_confusion_matrix(sample_data, test_dir):
    """Testa a plotagem da matriz de confusão"""
    y_true, y_pred, _ = sample_data
    save_path = os.path.join(test_dir, 'confusion_matrix.png')
    
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        classes=['Derrota', 'Empate', 'Vitória'],
        save_path=save_path,
        title='Teste Matriz de Confusão'
    )
    
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

@pytest.mark.visualization
def test_plot_roc_curves(sample_data, test_dir):
    """Testa a plotagem das curvas ROC"""
    y_true, _, y_prob = sample_data
    save_path = os.path.join(test_dir, 'roc_curves.png')
    
    plot_roc_curves(
        y_true=y_true,
        y_pred_proba=y_prob,
        classes=['Derrota', 'Empate', 'Vitória'],
        save_path=save_path,
        title='Teste Curvas ROC'
    )
    
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

@pytest.mark.visualization
def test_plot_feature_importance(test_dir):
    """Testa a plotagem da importância das features"""
    feature_importance = np.array([0.3, 0.2, 0.15, 0.25, 0.1])
    feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    save_path = os.path.join(test_dir, 'feature_importance.png')
    
    plot_feature_importance(
        feature_importance=feature_importance,
        feature_names=feature_names,
        save_path=save_path,
        title='Teste Importância das Features'
    )
    
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

@pytest.mark.visualization
def test_plot_prediction_distribution(test_dir):
    """Testa a plotagem da distribuição das predições"""
    predictions = {
        'Derrota': 0.3,
        'Empate': 0.2,
        'Vitória': 0.5
    }
    save_path = os.path.join(test_dir, 'prediction_distribution.png')
    
    plot_prediction_distribution(
        predictions=predictions,
        save_path=save_path,
        title='Teste Distribuição das Predições'
    )
    
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

@pytest.mark.visualization
def test_plot_model_comparison(test_dir):
    """Testa a plotagem da comparação entre modelos"""
    models_metrics = {
        'Random Forest': {'accuracy': 0.75, 'f1': 0.73},
        'XGBoost': {'accuracy': 0.78, 'f1': 0.76},
        'Neural Network': {'accuracy': 0.72, 'f1': 0.70}
    }
    save_path = os.path.join(test_dir, 'model_comparison.png')
    
    plot_model_comparison(
        models_metrics=models_metrics,
        metric_name='accuracy',
        save_path=save_path,
        title='Teste Comparação de Modelos'
    )
    
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

@pytest.mark.visualization
def test_plot_with_custom_style():
    """Testa a plotagem com estilo personalizado"""
    import matplotlib.pyplot as plt
    
    # Configurar estilo personalizado
    plt.style.use('seaborn')
    
    # Testar se o estilo foi aplicado
    assert plt.rcParams['axes.grid'] == True
    assert plt.rcParams['figure.figsize'] == [6.4, 4.8] 