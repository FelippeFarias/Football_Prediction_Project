import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, log_loss,
    brier_score_loss, precision_score, recall_score, f1_score
)
import logging

def evaluate_classification_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula métricas de classificação
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        y_prob: Probabilidades preditas (opcional)
    
    Returns:
        Dicionário com métricas
    """
    
    metrics = {}
    
    # Métricas básicas
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Métricas probabilísticas
    if y_prob is not None:
        # Converter y_true para one-hot
        n_classes = y_prob.shape[1]
        y_true_one_hot = np.zeros((len(y_true), n_classes))
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1
        
        # Calcular ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_one_hot, y_prob, multi_class='ovr')
        except ValueError:
            logging.warning("Não foi possível calcular ROC AUC")
        
        # Log loss e Brier score
        metrics['log_loss'] = log_loss(y_true, y_prob)
        metrics['brier_score'] = brier_score_loss(y_true_one_hot.ravel(), y_prob.ravel())
    
    return metrics

def evaluate_betting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    odds: Dict[str, float],
    threshold: float = 0.5
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Calcula métricas de apostas
    
    Args:
        y_true: Resultados reais
        y_pred: Predições
        y_prob: Probabilidades preditas
        odds: Odds para cada resultado
        threshold: Limite de probabilidade para apostas
        
    Returns:
        Dicionário com métricas por resultado
    """
    metrics = {}
    classes = ['Derrota', 'Empate', 'Vitória']
    
    for i, result in enumerate(classes):
        # Filtrar apostas acima do threshold
        mask = y_prob[:, i] >= threshold
        n_apostas = int(np.sum(mask))
        
        if n_apostas > 0:
            # Calcular precisão
            precisao = 100 * np.mean(y_true[mask] == i)
            
            # Calcular ROI esperado
            prob_media = np.mean(y_prob[mask, i])
            odd = odds[result]
            roi_esperado = 100 * (prob_media * odd - 1)
            
            # Calcular stake sugerido (Kelly)
            stake_sugerido = max(0, min(10, (prob_media * odd - 1) / (odd - 1)))
            
            metrics[result] = {
                'precisão': float(precisao),
                'roi_esperado': float(roi_esperado),
                'stake_sugerido': float(stake_sugerido),
                'n_apostas': n_apostas
            }
            
    return metrics

def evaluate_calibration(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Avalia calibração do modelo
    
    Args:
        y_true: Array com valores reais
        y_prob: Array com probabilidades preditas
        n_bins: Número de bins para calibração
        
    Returns:
        Tupla com (prob_true, prob_pred, counts)
    """
    # Garantir que temos dados suficientes
    if len(y_true) < n_bins * 2:
        n_bins = max(2, len(y_true) // 2)
        
    # Criar bins com número similar de amostras
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.percentile(y_prob, quantiles * 100)
    
    # Ajustar bins para garantir que são únicos
    bins = np.unique(bins)
    if len(bins) < 3:
        # Se temos poucos bins únicos, usar linspace
        bins = np.linspace(y_prob.min(), y_prob.max(), n_bins + 1)
    
    # Encontrar índices dos bins
    binids = np.digitize(y_prob, bins) - 1
    
    # Calcular médias das probabilidades preditas por bin
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins)-1)
    bin_counts = np.bincount(binids, minlength=len(bins)-1)
    bin_counts = np.maximum(bin_counts, 1)  # Evitar divisão por zero
    prob_pred = bin_sums / bin_counts
    
    # Calcular proporções reais por bin
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins)-1)
    prob_true = bin_true / bin_counts
    
    return prob_true, prob_pred, bin_counts

def evaluate_feature_importance(model: Any,
                              feature_names: List[str],
                              threshold: float = 0.01) -> Dict[str, float]:
    """
    Avalia importância das features
    
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

def evaluate_predictions_over_time(predictions: pd.DataFrame,
                               actual_results: pd.DataFrame,
                               window_size: int = 10) -> pd.DataFrame:
    """
    Avalia predições ao longo do tempo
    
    Args:
        predictions: DataFrame com predições
        actual_results: DataFrame com resultados reais
        window_size: Tamanho da janela móvel
        
    Returns:
        DataFrame com métricas ao longo do tempo
    """
    # Mesclar predições com resultados
    merged = predictions.merge(actual_results, on='game_id', suffixes=('_pred', '_true'))
    
    # Ordenar por data
    merged.sort_values('date', inplace=True)
    
    # Extrair probabilidades
    if isinstance(merged['probability'].iloc[0], (list, np.ndarray)):
        # Se probabilidade é array/lista
        merged['prob_0'] = merged['probability'].apply(lambda x: x[0])
        merged['prob_1'] = merged['probability'].apply(lambda x: x[1])
        merged['prob_2'] = merged['probability'].apply(lambda x: x[2])
    else:
        # Se probabilidade é escalar
        merged['prob_0'] = merged['probability']
        merged['prob_1'] = 0
        merged['prob_2'] = 1 - merged['probability']
    
    # Calcular métricas em janela móvel
    metrics = []
    for i in range(len(merged)):
        start_idx = max(0, i - window_size + 1)
        window_data = merged.iloc[start_idx:i+1]
        
        if len(window_data) > 0:
            # Calcular métricas
            y_true = window_data['result_true']
            y_pred = window_data['prediction']
            
            metrics.append({
                'date': merged['date'].iloc[i],
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'brier': brier_score_loss(y_true == 0, window_data['prob_0'])
            })
        
    return pd.DataFrame(metrics) 