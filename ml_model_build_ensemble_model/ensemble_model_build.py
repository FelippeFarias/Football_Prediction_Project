import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import pickle
import warnings
from typing import Dict, Any, Tuple, List

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definir diretório raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes')

def load_data(window_size: int = 10) -> pd.DataFrame:
    """Carrega os dados do arquivo mais recente"""
    try:
        df_files = [f for f in os.listdir(DATA_DIR) if f.endswith(f'_v2.txt')]
        if not df_files:
            raise FileNotFoundError("Nenhum arquivo de dados encontrado")
        
        df_file = max([f for f in df_files if f'{window_size}_v2.txt' in f], 
                     key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))
        
        with open(os.path.join(DATA_DIR, df_file), 'rb') as myFile:
            df = pickle.load(myFile)
            
        logging.info(f"Dados carregados com sucesso para janela de {window_size} jogos")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {str(e)}")
        raise

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, StandardScaler, LabelEncoder]:
    """Prepara os dados para treinamento"""
    try:
        # Remover colunas não necessárias para o modelo
        feature_cols = [col for col in df.columns if col not in ['Fixture ID', 'Result Indicator', 'Team', 'Game Date']]
        
        # Converter colunas para numérico
        X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        y = df['Result Indicator'].values
        
        # Preencher valores NaN com 0
        X = X.fillna(0)
        
        # Normalização e encoding
        scaler = StandardScaler()
        le = LabelEncoder()
        
        X = scaler.fit_transform(X)
        y = le.fit_transform(y)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Balanceamento com SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)
        
        return X_train_balanced, X_test, y_train_balanced, y_test, feature_cols, scaler, le
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def create_base_models() -> List[Tuple[str, Any]]:
    """Cria os modelos base do ensemble"""
    models = [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('xgb', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1
        )),
        ('lgb', LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ))
    ]
    
    return models

def create_ensemble(models: List[Tuple[str, Any]]) -> VotingClassifier:
    """Cria o modelo ensemble"""
    return VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )

def train_and_evaluate_models(X_train: np.ndarray,
                            X_test: np.ndarray,
                            y_train: np.ndarray,
                            y_test: np.ndarray,
                            models: List[Tuple[str, Any]],
                            ensemble: VotingClassifier) -> Dict[str, Any]:
    """Treina e avalia os modelos individuais e o ensemble"""
    results = {}
    
    # Treinar e avaliar modelos individuais
    for name, model in models:
        logging.info(f"\nTreinando {name}...")
        model.fit(X_train, y_train)
        
        # Avaliar modelo
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'report': report
        }
        
        logging.info(f"Score de treino ({name}): {train_score:.4f}")
        logging.info(f"Score de teste ({name}): {test_score:.4f}")
    
    # Treinar e avaliar ensemble
    logging.info("\nTreinando ensemble...")
    ensemble.fit(X_train, y_train)
    
    ensemble_train_score = ensemble.score(X_train, y_train)
    ensemble_test_score = ensemble.score(X_test, y_test)
    
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_report = classification_report(y_test, y_pred_ensemble)
    
    results['ensemble'] = {
        'model': ensemble,
        'train_score': ensemble_train_score,
        'test_score': ensemble_test_score,
        'report': ensemble_report
    }
    
    logging.info(f"Score de treino (ensemble): {ensemble_train_score:.4f}")
    logging.info(f"Score de teste (ensemble): {ensemble_test_score:.4f}")
    
    return results

def analyze_betting_performance(model: Any,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              le: LabelEncoder) -> Dict[str, float]:
    """Analisa o desempenho do modelo para apostas"""
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Converter para labels originais
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)
    
    betting_metrics = {}
    resultados = ['Derrota', 'Empate', 'Vitória']
    odds_medias = {'Derrota': 4.0, 'Empate': 3.5, 'Vitória': 2.5}
    
    for i, resultado in enumerate(resultados):
        precision = precision_recall_curve(y_test == i, y_pred_proba[:, i])[0].mean()
        
        if precision > 0.5:
            odds = odds_medias[resultado]
            roi = (precision * odds) - 1
            
            # Calcular stake sugerido baseado na confiança
            confianca = np.mean(y_pred_proba[:, i][y_pred == i])
            stake_sugerido = min(confianca * 10, 10)
            
            betting_metrics[resultado] = {
                'precisao': precision,
                'roi': roi,
                'stake_sugerido': stake_sugerido
            }
    
    return betting_metrics

def plot_results(results: Dict[str, Any],
                X_test: np.ndarray,
                y_test: np.ndarray,
                window_size: int):
    """Plota visualizações dos resultados"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'figures'), exist_ok=True)
    
    # Comparação de performance
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    train_scores = [results[m]['train_score'] for m in models]
    test_scores = [results[m]['test_score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Train')
    plt.bar(x + width/2, test_scores, width, label='Test')
    
    plt.xlabel('Modelos')
    plt.ylabel('Score')
    plt.title('Comparação de Performance dos Modelos')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'figures', f'model_comparison_{window_size}_{timestamp}.png'))
    plt.close()
    
    # ROC Curves para o ensemble
    ensemble = results['ensemble']['model']
    y_pred_proba = ensemble.predict_proba(X_test)
    
    plt.figure(figsize=(10, 6))
    for i, resultado in enumerate(['Derrota', 'Empate', 'Vitória']):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{resultado} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC do Ensemble')
    plt.legend()
    plt.savefig(os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'figures', f'roc_curves_{window_size}_{timestamp}.png'))
    plt.close()

def save_models(results: Dict[str, Any],
                scaler: StandardScaler,
                le: LabelEncoder,
                window_size: int):
    """Salva os modelos e transformadores"""
    os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'ml_models'), exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Salvar cada modelo
    for name, result in results.items():
        model_path = os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'ml_models', f'{name}_{window_size}_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
    
    # Salvar transformadores
    scaler_path = os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'ml_models', f'scaler_{window_size}_{timestamp}.pkl')
    le_path = os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'ml_models', f'label_encoder_{window_size}_{timestamp}.pkl')
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    logging.info(f"Modelos e transformadores salvos em ml_model_build_ensemble_model/ml_models/")

def main():
    print('\n\n ---------------- START ---------------- \n')
    
    try:
        # Criar diretórios necessários
        os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'ml_models'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_ensemble_model', 'figures'), exist_ok=True)
        
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            X_train, X_test, y_train, y_test, feature_cols, scaler, le = prepare_data(df)
            
            # Criar modelos
            base_models = create_base_models()
            ensemble = create_ensemble(base_models)
            
            # Treinar e avaliar modelos
            results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test,
                base_models, ensemble
            )
            
            # Análise de apostas para o ensemble
            betting_metrics = analyze_betting_performance(
                results['ensemble']['model'],
                X_test, y_test, le
            )
            
            logging.info("\nMétricas para apostas (Ensemble):")
            for resultado, metricas in betting_metrics.items():
                logging.info(f"\n{resultado}:")
                logging.info(f"Precisão: {metricas['precisao']:.2%}")
                logging.info(f"ROI esperado: {metricas['roi']:.2%}")
                logging.info(f"Stake sugerido: {metricas['stake_sugerido']:.1f}%")
            
            # Plotar resultados
            plot_results(results, X_test, y_test, window_size)
            
            # Salvar modelos
            save_models(results, scaler, le, window_size)
            
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main() 