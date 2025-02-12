import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import warnings
from typing import Dict, Any, Tuple

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print('\n\n ---------------- START ---------------- \n')

def load_data(window_size: int = 10) -> pd.DataFrame:
    """Carrega os dados do arquivo mais recente"""
    try:
        df_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') if f.endswith(f'_v2.txt')]
        if not df_files:
            raise FileNotFoundError("Nenhum arquivo de dados encontrado")
        
        df_file = max([f for f in df_files if f'{window_size}_v2.txt' in f], 
                     key=lambda x: os.path.getctime(os.path.join('prem_clean_fixtures_and_dataframes', x)))
        
        with open(f'prem_clean_fixtures_and_dataframes/{df_file}', 'rb') as myFile:
            df = pickle.load(myFile)
            
        logging.info(f"Dados carregados com sucesso para janela de {window_size} jogos")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {str(e)}")
        raise

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, StandardScaler, LabelEncoder]:
    """Prepara os dados para treinamento"""
    try:
        feature_cols = [col for col in df.columns if col not in ['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']]
        X = df[feature_cols].values
        y = df['Team Result Indicator'].values
        
        # Normalização e encoding
        scaler = StandardScaler()
        le = LabelEncoder()
        
        X = scaler.fit_transform(X)
        y = le.fit_transform(y)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_cols, scaler, le
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def optimize_knn(X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """Otimiza hiperparâmetros do KNN"""
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    logging.info(f"Melhores parâmetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_model(model: KNeighborsClassifier,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  window_size: int,
                  le: LabelEncoder) -> Dict[str, Any]:
    """Avalia o modelo com métricas detalhadas"""
    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Converter para labels originais
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)
    
    # Métricas básicas
    report = classification_report(y_test_original, y_pred_original)
    cm = confusion_matrix(y_test_original, y_pred_original)
    
    # ROC Curve para cada classe
    n_classes = len(np.unique(y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Análise de apostas
    resultados = ['Derrota', 'Empate', 'Vitória']
    odds_medias = {'Derrota': 4.0, 'Empate': 3.5, 'Vitória': 2.5}
    betting_metrics = {}
    
    for i, resultado in enumerate(resultados):
        precision = precision_recall_curve((y_test == i).astype(int), y_pred_proba[:, i])[0].mean()
        
        if precision > 0.5:
            odds = odds_medias[resultado]
            roi = (precision * odds) - 1
            
            confianca = np.mean(y_pred_proba[:, i][y_pred == i])
            stake_sugerido = min(confianca * 10, 10)
            
            betting_metrics[resultado] = {
                'precisao': precision,
                'roi': roi,
                'stake_sugerido': stake_sugerido
            }
    
    return {
        'predictions': y_pred_original,
        'probabilities': y_pred_proba,
        'report': report,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr, roc_auc),
        'betting_metrics': betting_metrics
    }

def plot_results(evaluation_results: Dict[str, Any], window_size: int):
    """Plota visualizações dos resultados"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('ml_model_build_nearest_neighbor/figures', exist_ok=True)
    
    # Matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        evaluation_results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Derrota', 'Empate', 'Vitória'],
        yticklabels=['Derrota', 'Empate', 'Vitória']
    )
    plt.title(f'Matriz de Confusão KNN - Janela de {window_size} Jogos')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig(f'ml_model_build_nearest_neighbor/figures/confusion_matrix_{window_size}_{timestamp}.png')
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red', 'green']
    labels = ['Derrota', 'Empate', 'Vitória']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(
            evaluation_results['roc_data'][0][i],
            evaluation_results['roc_data'][1][i],
            color=color,
            label=f'{label} (AUC = {evaluation_results["roc_data"][2][i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curvas ROC KNN - Janela de {window_size} Jogos')
    plt.legend(loc='lower right')
    plt.savefig(f'ml_model_build_nearest_neighbor/figures/roc_curves_{window_size}_{timestamp}.png')
    plt.close()

def save_model(model: KNeighborsClassifier,
              scaler: StandardScaler,
              le: LabelEncoder,
              window_size: int):
    """Salva o modelo e os transformadores"""
    os.makedirs('ml_model_build_nearest_neighbor/ml_models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = f'ml_model_build_nearest_neighbor/ml_models/knn_{window_size}_{timestamp}.pkl'
    scaler_path = f'ml_model_build_nearest_neighbor/ml_models/scaler_{window_size}_{timestamp}.pkl'
    le_path = f'ml_model_build_nearest_neighbor/ml_models/label_encoder_{window_size}_{timestamp}.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    logging.info(f"Modelo e transformadores salvos em {model_path}")

def main():
    try:
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            X_train, X_test, y_train, y_test, feature_cols, scaler, le = prepare_data(df)
            
            # Otimizar e treinar modelo
            model = optimize_knn(X_train, y_train)
            
            # Avaliar modelo
            evaluation_results = evaluate_model(model, X_test, y_test, window_size, le)
            
            # Logging dos resultados
            logging.info(f"\nRelatório de Classificação:\n{evaluation_results['report']}")
            
            # Análise de apostas
            logging.info("\nMétricas para apostas:")
            for resultado, metricas in evaluation_results['betting_metrics'].items():
                logging.info(f"\n{resultado}:")
                logging.info(f"Precisão: {metricas['precisao']:.2%}")
                logging.info(f"ROI esperado: {metricas['roi']:.2%}")
                logging.info(f"Stake sugerido: {metricas['stake_sugerido']:.1f}%")
            
            # Plotar resultados
            plot_results(evaluation_results, window_size)
            
            # Salvar modelo
            save_model(model, scaler, le, window_size)
            
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main() 