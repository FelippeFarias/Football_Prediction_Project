import os
import glob
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_latest_data(data_dir: str,
                    pattern: str = '*.csv') -> Tuple[pd.DataFrame, str]:
    """
    Carrega dados do arquivo CSV mais recente

    Args:
        data_dir: Diretório com arquivos
        pattern: Padrão para busca de arquivos

    Returns:
        DataFrame com dados e nome do arquivo
    """
    # Listar arquivos
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {data_dir} com padrão {pattern}")

    # Encontrar arquivo mais recente
    latest_file = max(files, key=os.path.getctime)

    # Carregar dados
    df = pd.read_csv(latest_file)

    # Converter tipos numéricos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Manter tipo original (int32/int64/float32/float64)
        if df[col].dtype == np.int32:
            df[col] = df[col].astype(np.int32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int64)
        elif df[col].dtype == np.float32:
            df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype(np.float64)

    return df, os.path.basename(latest_file)

def save_model(model: Any,
              scaler: StandardScaler,
              label_encoder: LabelEncoder,
              save_dir: str,
              window_size: int = 10) -> None:
    """
    Salva modelo, scaler e label encoder

    Args:
        model: Modelo treinado
        scaler: Scaler ajustado
        label_encoder: Label encoder ajustado
        save_dir: Diretório para salvar
        window_size: Tamanho da janela usada
    """
    # Criar diretório se não existir
    os.makedirs(save_dir, exist_ok=True)

    # Gerar timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Salvar modelo
    model_path = os.path.join(save_dir, f'model_{timestamp}_w{window_size}.joblib')
    joblib.dump(model, model_path)

    # Salvar scaler
    scaler_path = os.path.join(save_dir, f'scaler_{timestamp}_w{window_size}.joblib')
    joblib.dump(scaler, scaler_path)

    # Salvar label encoder
    le_path = os.path.join(save_dir, f'label_encoder_{timestamp}_w{window_size}.joblib')
    joblib.dump(label_encoder, le_path)

def load_model(model_dir: str,
              window_size: int = 10) -> Tuple[Any, StandardScaler, LabelEncoder]:
    """
    Carrega modelo e transformadores

    Args:
        model_dir: Diretório com arquivos
        window_size: Tamanho da janela usada

    Returns:
        Modelo, scaler e label encoder
    """
    # Encontrar arquivos mais recentes
    model_files = glob.glob(os.path.join(model_dir, f'model_*_w{window_size}.joblib'))
    scaler_files = glob.glob(os.path.join(model_dir, f'scaler_*_w{window_size}.joblib'))
    le_files = glob.glob(os.path.join(model_dir, f'label_encoder_*_w{window_size}.joblib'))

    if not (model_files and scaler_files and le_files):
        raise FileNotFoundError(f"Arquivos não encontrados em {model_dir}")

    # Carregar arquivos mais recentes
    model = joblib.load(max(model_files, key=os.path.getctime))
    scaler = joblib.load(max(scaler_files, key=os.path.getctime))
    label_encoder = joblib.load(max(le_files, key=os.path.getctime))

    return model, scaler, label_encoder

def format_predictions(y_pred: np.ndarray,
                      y_prob: np.ndarray,
                      label_encoder: LabelEncoder,
                      df: pd.DataFrame) -> pd.DataFrame:
    """
    Formata predições
    
    Args:
        y_pred: Predições
        y_prob: Probabilidades
        label_encoder: Label encoder
        df: DataFrame original
        
    Returns:
        DataFrame com predições formatadas
    """
    # Criar DataFrame
    pred_df = df.copy()
    
    # Adicionar predições
    pred_df['prediction'] = label_encoder.inverse_transform(y_pred)
    
    # Adicionar probabilidades
    for i, class_name in enumerate(label_encoder.classes_):
        pred_df[f'prob_{i}'] = y_prob[:, i]
        
    return pred_df

def get_betting_advice(pred_df: pd.DataFrame,
                      threshold: float = 0.6,
                      bankroll: float = 1000.0) -> List[Dict[str, Any]]:
    """
    Gera recomendações de apostas
    
    Args:
        pred_df: DataFrame com predições
        threshold: Limite de probabilidade
        bankroll: Valor disponível
        
    Returns:
        Lista com recomendações
    """
    advice = []
    
    for _, row in pred_df.iterrows():
        # Verificar probabilidades
        probs = {
            'home_win': row['prob_home_win'],
            'draw': row['prob_draw'],
            'away_win': row['prob_away_win']
        }
        
        odds = {
            'home_win': row['odds_home'],
            'draw': row['odds_draw'],
            'away_win': row['odds_away']
        }
        
        for bet_type, prob in probs.items():
            if prob >= threshold:
                # Calcular stake (Kelly)
                odd = odds[bet_type]
                stake = max(0, min(0.1 * bankroll, 
                                 bankroll * (prob * odd - 1) / (odd - 1)))
                
                advice.append({
                    'match': f"{row['home_team']} vs {row['away_team']}",
                    'bet_type': bet_type,
                    'probability': prob,
                    'odds': odd,
                    'stake': stake
                })
                
    return advice

def calculate_stake(prob: float,
                  odds: float,
                  bankroll: float,
                  max_stake: float = 0.1) -> float:
    """
    Calcula o stake usando o critério de Kelly

    Args:
        prob: Probabilidade estimada
        odds: Odds decimal
        bankroll: Bankroll total
        max_stake: Stake máximo como fração do bankroll

    Returns:
        Valor do stake recomendado
    """
    # Validar inputs
    if not 0 <= prob <= 1:
        raise ValueError("Probabilidade deve estar entre 0 e 1")
    if odds <= 1:
        raise ValueError("Odds deve ser maior que 1")
    if bankroll <= 0:
        raise ValueError("Bankroll deve ser positivo")
    if not 0 < max_stake <= 1:
        raise ValueError("max_stake deve estar entre 0 e 1")

    # Calcular fração de Kelly
    q = 1 - prob
    b = odds - 1
    f = (b * prob - q) / b

    # Limitar stake
    f = max(0, min(f, max_stake))

    # Calcular valor do stake
    stake = f * bankroll

    return stake 