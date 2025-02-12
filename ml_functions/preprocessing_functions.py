import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import stats

def prepare_features(df: pd.DataFrame,
                     feature_cols: List[str],
                     target_col: str,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Prepara features para treinamento
    
    Args:
        df: DataFrame com dados
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna alvo
        test_size: Proporção do conjunto de teste
        random_state: Semente aleatória
        
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    # Separar features e target
    X = df[feature_cols].values
    y = df[target_col].astype(str).values
    
    # Codificar target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Verificar se cada classe tem pelo menos 2 amostras
    unique_classes, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        # Duplicar amostras de classes com apenas 1 exemplo
        for class_idx in unique_classes[counts < 2]:
            sample_idx = np.where(y == class_idx)[0][0]
            X = np.vstack([X, X[sample_idx]])
            y = np.append(y, class_idx)
    
    # Normalizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, le

def handle_missing_values(df: pd.DataFrame,
                        numeric_strategy: str = 'mean',
                        categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Trata valores faltantes em features numéricas e categóricas

    Args:
        df: DataFrame com dados
        numeric_strategy: Estratégia para valores numéricos ('mean', 'median', 'zero')
        categorical_strategy: Estratégia para categóricos ('mode', 'missing')

    Returns:
        DataFrame com valores faltantes tratados
    """
    df_clean = df.copy()

    # Separar colunas por tipo
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    # Tratar numéricos
    if numeric_cols.any():
        if numeric_strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif numeric_strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif numeric_strategy == 'zero':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    # Tratar categóricos
    if categorical_cols.any():
        if categorical_strategy == 'mode':
            for col in categorical_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif categorical_strategy == 'missing':
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna('missing')

    return df_clean

def remove_outliers(df: pd.DataFrame,
                   cols: List[str],
                   n_std: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers usando z-score
    
    Args:
        df: DataFrame com dados
        cols: Lista de colunas para remover outliers
        n_std: Número de desvios padrão
    
    Returns:
        DataFrame sem outliers
    """
    df_clean = df.copy()
    
    for col in cols:
        # Calcular z-score
        z_scores = stats.zscore(df_clean[col])
        
        # Remover outliers
        df_clean = df_clean[abs(z_scores) <= n_std]
    
    return df_clean

def encode_categorical_features(df: pd.DataFrame,
                              cat_cols: List[str],
                              encoding: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder]]]:
    """
    Codifica features categóricas
    
    Args:
        df: DataFrame com dados
        cat_cols: Lista de colunas categóricas
        encoding: Tipo de codificação ('label' ou 'onehot')
    
    Returns:
        DataFrame codificado e dicionário com encoders
    """
    df_encoded = df.copy()
    encoders = {}
    
    if encoding == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            df_encoded[col] = df_encoded[col].astype('int64')
            encoders[col] = le
            
    elif encoding == 'onehot':
        for col in cat_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = ohe.fit_transform(df_encoded[[col]])
            feature_names = [f"{col}_{val}" for val in ohe.categories_[0]]
            
            # Adicionar colunas one-hot
            for i, name in enumerate(feature_names):
                df_encoded[name] = encoded[:, i]
            
            # Remover coluna original
            df_encoded = df_encoded.drop(col, axis=1)
            encoders[col] = ohe
            
    return df_encoded, encoders

def scale_features(df: pd.DataFrame,
                  num_cols: List[str],
                  scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Escala features numéricas
    
    Args:
        df: DataFrame com dados
        num_cols: Lista de colunas numéricas
        scaler: Scaler pré-treinado (opcional)
    
    Returns:
        DataFrame escalado e scaler
    """
    df_scaled = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    else:
        df_scaled[num_cols] = scaler.transform(df_scaled[num_cols])
    
    return df_scaled, scaler

def create_time_features(df: pd.DataFrame,
                        date_col: str) -> pd.DataFrame:
    """
    Cria features temporais a partir de coluna de data

    Args:
        df: DataFrame com dados
        date_col: Nome da coluna de data

    Returns:
        DataFrame com features temporais
    """
    df_time = df.copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col])

    # Extrair features
    df_time['year'] = df_time[date_col].dt.year.astype('int64')
    df_time['month'] = df_time[date_col].dt.month.astype('int64')
    df_time['day'] = df_time[date_col].dt.day.astype('int64')
    df_time['dayofweek'] = df_time[date_col].dt.dayofweek.astype('int64')
    df_time['quarter'] = df_time[date_col].dt.quarter.astype('int64')
    df_time['is_weekend'] = (df_time['dayofweek'] >= 5).astype('int64')

    # Remover coluna original
    df_time = df_time.drop(date_col, axis=1)

    return df_time

def create_interaction_features(df: pd.DataFrame,
                              feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Cria features de interação
    
    Args:
        df: DataFrame com dados
        feature_pairs: Lista de tuplas com pares de features
    
    Returns:
        DataFrame com features de interação
    """
    
    df_interact = df.copy()
    
    for feat1, feat2 in feature_pairs:
        # Criar nome da nova feature
        new_feat = f"{feat1}_{feat2}_interaction"
        
        # Calcular interação
        df_interact[new_feat] = df_interact[feat1] * df_interact[feat2]
    
    return df_interact 