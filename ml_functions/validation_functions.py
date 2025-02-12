import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

def cross_validate_model(model: Any,
                        X: np.ndarray,
                        y: np.ndarray,
                        cv_method: str = 'stratified',
                        n_splits: int = 5,
                        random_state: int = 42) -> Dict[str, List[float]]:
    """
    Realiza validação cruzada do modelo
    
    Args:
        model: Modelo a ser validado
        X: Features
        y: Target
        cv_method: Método de CV ('stratified', 'kfold' ou 'timeseries')
        n_splits: Número de folds
        random_state: Seed para reprodutibilidade
    
    Returns:
        Dicionário com métricas por fold
    """
    
    # Inicializar dicionário de métricas
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Selecionar método de CV
    if cv_method == 'stratified':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_method == 'kfold':
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_method == 'timeseries':
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"Método de CV inválido: {cv_method}")
    
    # Realizar CV
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Separar dados
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_val)
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        # Armazenar métricas
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        
        logging.info(f"Fold {fold+1}/{n_splits}:")
        logging.info(f"  Acurácia: {accuracy:.3f}")
        logging.info(f"  Precisão: {precision:.3f}")
        logging.info(f"  Recall: {recall:.3f}")
        logging.info(f"  F1: {f1:.3f}\n")
    
    return metrics

def validate_predictions(predictions: pd.DataFrame,
                        actual_results: pd.DataFrame,
                        threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Valida predições históricas
    
    Args:
        predictions: DataFrame com predições
        actual_results: DataFrame com resultados reais
        threshold: Limiar de confiança
    
    Returns:
        Dicionário com métricas de validação
    """
    
    metrics = {}
    
    # Mesclar predições com resultados
    merged = pd.merge(predictions, actual_results, on='game_id', how='inner')
    
    # Validar cada tipo de aposta
    for bet_type in ['home_win', 'draw', 'away_win']:
        # Filtrar apostas com alta confiança
        confident_bets = merged[merged[f'{bet_type}_prob'] >= threshold]
        
        if len(confident_bets) == 0:
            continue
        
        # Calcular métricas
        correct = (confident_bets[f'{bet_type}_pred'] == confident_bets['result']).mean()
        
        # Calcular ROI
        if bet_type == 'draw':
            odds = 3.5
        else:
            odds = 2.5
        
        roi = (correct * odds - 1) * 100
        
        metrics[bet_type] = {
            'precisão': correct * 100,
            'roi': roi,
            'n_apostas': len(confident_bets)
        }
    
    return metrics

def validate_model_stability(model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           n_iterations: int = 10,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Valida estabilidade do modelo
    
    Args:
        model: Modelo a ser validado
        X: Features
        y: Target
        n_iterations: Número de iterações
        test_size: Tamanho do conjunto de teste
        random_state: Seed para reprodutibilidade
    
    Returns:
        Dicionário com métricas de estabilidade
    """
    
    # Inicializar listas para métricas
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    # Realizar iterações
    for i in range(n_iterations):
        # Dividir dados
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state + i,
            stratify=y
        )
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        accuracies.append(accuracy_score(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Calcular estatísticas
    metrics = {
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        },
        'precision': {
            'mean': np.mean(precisions),
            'std': np.std(precisions),
            'min': np.min(precisions),
            'max': np.max(precisions)
        },
        'recall': {
            'mean': np.mean(recalls),
            'std': np.std(recalls),
            'min': np.min(recalls),
            'max': np.max(recalls)
        },
        'f1': {
            'mean': np.mean(f1s),
            'std': np.std(f1s),
            'min': np.min(f1s),
            'max': np.max(f1s)
        }
    }
    
    return metrics

def validate_feature_importance(model: Any,
                              feature_names: List[str],
                              threshold: float = 0.01) -> Dict[str, float]:
    """
    Valida importância das features
    
    Args:
        model: Modelo treinado
        feature_names: Lista com nomes das features
        threshold: Limiar de importância
    
    Returns:
        Dicionário com importância das features
    """
    
    # Verificar se modelo tem atributo feature_importances_
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Modelo não possui atributo feature_importances_")
    
    # Criar dicionário de importâncias
    importances = {}
    for name, importance in zip(feature_names, model.feature_importances_):
        if importance >= threshold:
            importances[name] = importance
    
    # Ordenar por importância
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances 