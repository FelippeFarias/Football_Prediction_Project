import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

def optimize_model(model: Any,
                  param_grid: Dict[str, List[Any]],
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  cv: int = 5,
                  scoring: str = 'accuracy',
                  n_jobs: int = -1) -> Tuple[Any, Dict[str, Any]]:
    """
    Otimiza hiperparâmetros do modelo
    
    Args:
        model: Modelo base
        param_grid: Grid de parâmetros
        X_train: Features de treino
        y_train: Target de treino
        cv: Número de folds
        scoring: Métrica para otimização
        n_jobs: Número de jobs paralelos
    
    Returns:
        Tupla com (modelo otimizado, melhores parâmetros)
    """
    
    # Configurar scoring personalizado se necessário
    if scoring == 'custom':
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(lambda y_true, y_pred: 
                precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]),
            'recall': make_scorer(lambda y_true, y_pred:
                precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]),
            'f1': make_scorer(lambda y_true, y_pred:
                precision_recall_fscore_support(y_true, y_pred, average='weighted')[2])
        }
        refit = 'f1'
    else:
        refit = True
    
    # Criar e ajustar GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Logar resultados
    logging.info("Resultados da otimização:")
    logging.info(f"Melhores parâmetros: {grid_search.best_params_}")
    logging.info(f"Melhor score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_model(model: Any,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray) -> Tuple[Any, Dict[str, float]]:
    """
    Treina um modelo e retorna métricas

    Args:
        model: Modelo base para treinar
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação

    Returns:
        Modelo treinado e dicionário de métricas
    """
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Calcular métricas de treino
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Calcular métricas de validação
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='weighted')
    recall = recall_score(y_val, y_val_pred, average='weighted')
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return model, metrics

def predict_proba_calibrated(model: Any,
                           X: np.ndarray,
                           calibration_method: str = 'isotonic') -> np.ndarray:
    """
    Calibra probabilidades do modelo

    Args:
        model: Modelo base
        X: Features para predição
        calibration_method: Método de calibração ('isotonic' ou 'sigmoid')

    Returns:
        Array com probabilidades calibradas
    """
    
    # Criar e treinar calibrador
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=calibration_method,
        cv='prefit'
    )
    
    # Obter predições originais
    y_pred = model.predict(X)
    
    # Treinar calibrador com as mesmas features
    calibrated_model.fit(X, y_pred)
    
    # Retornar probabilidades calibradas
    return calibrated_model.predict_proba(X)

def ensemble_predictions(models: List[Any],
                        X: np.ndarray,
                        weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combina predições de múltiplos modelos

    Args:
        models: Lista de modelos treinados
        X: Features para predição
        weights: Pesos para cada modelo

    Returns:
        Tupla com (predições, probabilidades)
    """
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Calcular probabilidades ponderadas
    probas = np.zeros((X.shape[0], 3))  # 3 classes
    for model, weight in zip(models, weights):
        probas += weight * model.predict_proba(X)
    
    # Normalizar probabilidades
    probas /= np.sum(weights)
    
    # Obter predições
    predictions = np.argmax(probas, axis=1)
    
    return predictions, probas

def get_model_explanation(model: Any,
                         X: np.ndarray,
                         feature_names: List[str],
                         n_features: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Gera explicação do modelo

    Args:
        model: Modelo treinado
        X: Features para explicação
        feature_names: Nomes das features
        n_features: Número de features para mostrar

    Returns:
        Dicionário com importâncias por classe
    """
    
    if hasattr(model, 'feature_importances_'):
        # Para modelos baseados em árvores
        importances = model.feature_importances_
    else:
        # Para outros modelos, usar permutation importance
        importances = permutation_importance(
            model, X, model.predict(X), n_repeats=10
        ).importances_mean
    
    # Criar dicionário de importâncias
    feature_importance = {}
    for i, (name, importance) in enumerate(zip(feature_names, importances)):
        if i < n_features:
            feature_importance[name] = float(importance)
    
    return {
        'feature_importance': feature_importance,
        'n_features_shown': min(n_features, len(feature_names))
    } 