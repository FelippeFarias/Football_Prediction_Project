import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from .utils import load_model, format_predictions
from .evaluation_functions import evaluate_betting_metrics
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def prepare_prediction_data(df: pd.DataFrame,
                          scaler: StandardScaler,
                          feature_cols: List[str]) -> np.ndarray:
    """
    Prepara dados para predição
    
    Args:
        df: DataFrame com dados
        scaler: Scaler ajustado
        feature_cols: Lista de colunas de features
        
    Returns:
        Array com dados preparados
    """
    # Selecionar features
    X = df[feature_cols].copy()
    
    # Aplicar scaling
    X_scaled = scaler.transform(X)
    
    return X_scaled

def make_predictions(model: BaseEstimator,
                    X: np.ndarray,
                    scaler: StandardScaler,
                    label_encoder: LabelEncoder,
                    df: pd.DataFrame,
                    threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Faz predições usando modelo treinado
    
    Args:
        model: Modelo treinado
        X: Features para predição
        scaler: Scaler ajustado
        label_encoder: Label encoder ajustado
        df: DataFrame original
        threshold: Limiar de confiança
        
    Returns:
        Tupla com (DataFrame de predições, métricas de apostas)
    """
    # Preparar dados
    X_scaled = scaler.transform(X)
    
    # Fazer predições
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)
    
    # Formatar predições
    pred_df = format_predictions(y_pred, y_prob, label_encoder, df)
    
    # Calcular métricas de apostas
    odds = {
        'Derrota': 2.5,
        'Empate': 3.5,
        'Vitória': 2.5
    }
    
    if 'Result Indicator' in df.columns:
        y_true = label_encoder.transform(df['Result Indicator'])
        betting_metrics = evaluate_betting_metrics(y_true, y_pred, y_prob, odds, threshold)
    else:
        betting_metrics = None
    
    return pred_df, betting_metrics

def predict_match(model: BaseEstimator,
                 scaler: StandardScaler,
                 label_encoder: LabelEncoder,
                 match_data: pd.DataFrame,
                 feature_cols: List[str]) -> Dict[str, Any]:
    """
    Prediz resultado de uma partida
    
    Args:
        model: Modelo treinado
        scaler: Scaler ajustado
        label_encoder: Label encoder ajustado
        match_data: DataFrame com dados da partida
        feature_cols: Lista de colunas de features
        
    Returns:
        Dicionário com predições
    """
    # Preparar dados
    X = prepare_prediction_data(match_data, scaler, feature_cols)
    
    # Fazer predições
    predictions = make_predictions(model, X, scaler, label_encoder, match_data)
    
    # Formatar resultados
    formatted_predictions = format_predictions(
        y_pred=label_encoder.transform(predictions[0]['prediction']),
        y_prob=np.array([predictions[0]['probabilities'][class_name]
                        for class_name in label_encoder.classes_]).T,
        label_encoder=label_encoder,
        df=match_data
    )
    
    return formatted_predictions

def predict_next_games(model_dir: str,
                      data_dir: str,
                      window_size: int = 10,
                      threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Prediz próximos jogos
    
    Args:
        model_dir: Diretório com modelo
        data_dir: Diretório com dados
        window_size: Tamanho da janela
        threshold: Limiar de confiança
        
    Returns:
        Tupla com (DataFrame de predições, métricas de apostas)
    """
    # Carregar modelo e transformadores
    model, scaler, label_encoder = load_model(model_dir, f'model_{window_size}')
    
    # Carregar dados
    df = pd.read_csv(os.path.join(data_dir, 'next_games.csv'))
    
    # Preparar features
    feature_cols = [col for col in df.columns if col not in ['Result Indicator', 'Date']]
    X = df[feature_cols]
    
    # Fazer predições
    pred_df, betting_metrics = make_predictions(
        model, X, scaler, label_encoder, df, threshold
    )
    
    return pred_df, betting_metrics

def predict_with_ensemble(models: List[Any],
                         X: np.ndarray,
                         weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faz predições usando ensemble de modelos
    
    Args:
        models: Lista de modelos
        X: Features para predição
        weights: Pesos para cada modelo (opcional)
    
    Returns:
        Tupla com (labels preditos, probabilidades)
    """
    
    # Usar pesos iguais se não especificados
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    # Normalizar pesos
    weights = np.array(weights) / sum(weights)
    
    # Fazer predições com cada modelo
    probas = []
    for model in models:
        if hasattr(model, 'predict_proba'):
            probas.append(model.predict_proba(X))
        else:
            # Converter predict para predict_proba
            y_pred = model.predict(X)
            n_classes = len(np.unique(y_pred))
            proba = np.zeros((len(y_pred), n_classes))
            for i, pred in enumerate(y_pred):
                proba[i, pred] = 1
            probas.append(proba)
    
    # Combinar probabilidades
    weighted_probas = np.zeros_like(probas[0])
    for weight, proba in zip(weights, probas):
        weighted_probas += weight * proba
    
    # Obter labels preditos
    y_pred = np.argmax(weighted_probas, axis=1)
    
    return y_pred, weighted_probas

def predict_with_calibration(model: Any,
                           X: np.ndarray,
                           calibration_method: str = 'isotonic') -> np.ndarray:
    """
    Faz predições com probabilidades calibradas
    
    Args:
        model: Modelo treinado
        X: Features para predição
        calibration_method: Método de calibração
    
    Returns:
        Array com probabilidades calibradas
    """
    
    from sklearn.calibration import CalibratedClassifierCV
    
    # Verificar se modelo já está calibrado
    if hasattr(model, '_calibrated') and model._calibrated:
        return model.predict_proba(X)
    
    # Calibrar modelo
    calibrated_model = CalibratedClassifierCV(
        model,
        cv='prefit',
        method=calibration_method
    )
    
    # Marcar como calibrado
    calibrated_model._calibrated = True
    
    return calibrated_model.predict_proba(X)

def get_prediction_explanation(model: Any,
                             X: np.ndarray,
                             feature_names: List[str],
                             prediction_idx: int,
                             n_features: int = 5) -> Dict[str, Any]:
    """
    Gera explicação para uma predição específica
    
    Args:
        model: Modelo treinado
        X: Features para explicação
        feature_names: Nomes das features
        prediction_idx: Índice da predição
        n_features: Número de features importantes
    
    Returns:
        Dicionário com explicação
    """
    
    explanation = {}
    
    # Verificar importância local das features
    if hasattr(model, 'feature_importances_'):
        # Pegar valores das features
        feature_values = X[prediction_idx]
        
        # Ordenar features por importância
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Selecionar top features
        top_features = []
        for i in indices[:n_features]:
            top_features.append({
                'feature': feature_names[i],
                'importance': importances[i],
                'value': feature_values[i]
            })
        
        explanation['feature_contribution'] = top_features
    
    # Verificar coeficientes para modelos lineares
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) == 1:
            coef = coef.reshape(1, -1)
        
        # Pegar valores das features
        feature_values = X[prediction_idx]
        
        # Calcular contribuições
        contributions = coef * feature_values
        
        # Ordenar features por magnitude da contribuição
        indices = np.argsort(np.abs(contributions).ravel())[::-1]
        
        # Selecionar top features
        top_features = []
        for i in indices[:n_features]:
            top_features.append({
                'feature': feature_names[i],
                'coefficient': coef.ravel()[i],
                'value': feature_values[i],
                'contribution': contributions.ravel()[i]
            })
        
        explanation['feature_contribution'] = top_features
    
    return explanation 