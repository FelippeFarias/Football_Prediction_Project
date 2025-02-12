import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score
import pandas as pd

def calculate_betting_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_prob: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas para apostas
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        y_prob: Probabilidades preditas
        threshold: Limite de probabilidade para apostas
    
    Returns:
        Dicionário com métricas por tipo de resultado
    """
    
    metrics = {}
    
    # Para cada classe (0: Home, 1: Draw, 2: Away)
    for i in range(3):
        result_type = ['home', 'draw', 'away'][i]
        
        # Filtrar apostas acima do threshold
        mask = y_prob[:, i] >= threshold
        if np.sum(mask) > 0:
            precision = precision_score(y_true[mask], y_pred[mask], average='weighted', zero_division=0)
            n_bets = int(np.sum(mask))
            
            metrics[result_type] = {
                'precisão': float(precision * 100),
                'roi_esperado': float((precision * 2.0 - 1) * 100),
                'stake_sugerido': float(min(precision * 10, 10)),
                'n_apostas': n_bets
            }
        else:
            metrics[result_type] = {
                'precisão': 0.0,
                'roi_esperado': 0.0,
                'stake_sugerido': 0.0,
                'n_apostas': 0
            }
    
    return metrics

def calculate_model_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas gerais do modelo
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        y_prob: Probabilidades preditas
    
    Returns:
        Dicionário com métricas
    """
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Calcular log loss
    y_true_one_hot = np.eye(3)[y_true]
    log_loss_value = -np.mean(np.sum(y_true_one_hot * np.log(np.clip(y_prob, 1e-15, 1-1e-15)), axis=1))
    
    # Calcular Brier Score
    brier_score = np.mean(np.sum((y_true_one_hot - y_prob) ** 2, axis=1))
    
    return {
        'acurácia': float(accuracy * 100),
        'precisão': float(precision * 100),
        'recall': float(recall * 100),
        'f1': float(f1 * 100),
        'log_loss': float(log_loss_value),
        'brier_score': float(brier_score)
    }

def calculate_calibration_metrics(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula métricas de calibração
    
    Args:
        y_true: Labels verdadeiros
        y_prob: Probabilidades preditas
        n_bins: Número de bins para calibração
    
    Returns:
        Tupla com (prob_true, prob_pred, counts)
    """
    
    y_true_one_hot = np.eye(3)[y_true]
    
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(3):  # Para cada classe
        bin_edges = np.linspace(0., 1. + 1e-8, n_bins + 1)
        
        for j, (bin_lower, bin_upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (y_prob[:, i] > bin_lower) & (y_prob[:, i] <= bin_upper)
            if np.sum(mask) > 0:
                prob_true[j] += np.mean(y_true_one_hot[mask, i])
                prob_pred[j] += np.mean(y_prob[mask, i])
                counts[j] += np.sum(mask)
    
    # Normalizar
    non_zero = counts > 0
    prob_true[non_zero] /= 3
    prob_pred[non_zero] /= 3
    
    return prob_true, prob_pred, counts

def evaluate_predictions(predictions: pd.DataFrame,
                        actual_results: pd.DataFrame,
                        confidence_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Avalia predições históricas
    
    Args:
        predictions: DataFrame com predições
        actual_results: DataFrame com resultados reais
        confidence_threshold: Limiar de confiança
    
    Returns:
        Dicionário com métricas de avaliação
    """
    
    metrics = {}
    
    # Mesclar predições com resultados
    merged = pd.merge(predictions, actual_results, on='game_id', how='inner')
    
    # Calcular métricas para cada tipo de aposta
    for bet_type in ['home_win', 'draw', 'away_win']:
        # Filtrar apostas com alta confiança
        confident_bets = merged[merged[f'{bet_type}_prob'] >= confidence_threshold]
        
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