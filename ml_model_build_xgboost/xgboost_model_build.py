import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import joblib
import warnings

# Ignorar avisos específicos do XGBoost
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(window_size=10):
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

def prepare_data(df):
    """Prepara os dados com normalização e balanceamento avançado"""
    try:
        # Separar features e target
        feature_cols = [col for col in df.columns if col not in ['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']]
        X = df[feature_cols]
        y = df['Team Result Indicator']
        
        logging.info(f"Shape dos dados originais: X={X.shape}, y={y.shape}")
        
        # Label Encoding para o target
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Divisão treino/teste estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info(f"Shape após split: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Balanceamento com SMOTEENN
        smote_enn = SMOTEENN(random_state=42, sampling_strategy='auto')
        X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train)
        
        logging.info(f"Shape após balanceamento: X_train={X_train_balanced.shape}")
        
        # Converter para DataFrame mantendo nomes das colunas
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        # Verificar valores ausentes
        if X_train_balanced.isnull().any().any() or X_test_scaled.isnull().any().any():
            logging.warning("Detectados valores ausentes nos dados!")
            
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test, feature_cols, scaler, le
        
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train):
    """Otimiza hiperparâmetros do XGBoost com foco em métricas de apostas"""
    try:
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [4, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1],
            'gamma': [0]
        }
        
        logging.info("Iniciando otimização de hiperparâmetros...")
        logging.info(f"Shape dos dados de treino: X={X_train.shape}, y={y_train.shape}")
        
        xgb = XGBClassifier(
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logging.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        logging.info(f"Melhor score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
        
    except Exception as e:
        logging.error(f"Erro na otimização de hiperparâmetros: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, window_size, le):
    """Avalia o modelo com métricas específicas para apostas"""
    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Converter labels numéricos para nomes originais
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)
    
    # Métricas básicas
    logging.info(f"\nResultados para janela de {window_size} jogos:")
    logging.info(f"\nRelatório de classificação:\n{classification_report(y_test_original, y_pred_original)}")
    
    # Métricas específicas para apostas
    resultados = ['Derrota', 'Empate', 'Vitória']
    odds_medias = {'Derrota': 4.0, 'Empate': 3.5, 'Vitória': 2.5}
    
    for i, resultado in enumerate(resultados):
        precision = precision_recall_curve(y_test == i, y_pred_proba[:, i])[0]
        recall = precision_recall_curve(y_test == i, y_pred_proba[:, i])[1]
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logging.info(f"\nMétricas detalhadas para {resultado}:")
        logging.info(f"Precisão média: {np.mean(precision):.3f}")
        logging.info(f"Recall médio: {np.mean(recall):.3f}")
        logging.info(f"F1-Score médio: {np.mean(f1):.3f}")
        
        # Análise de ROI
        if np.mean(precision) > 0.5:
            odds = odds_medias[resultado]
            roi = (np.mean(precision) * odds) - 1
            logging.info(f"ROI esperado com odds {odds}: {roi:.2%}")
            
            # Sugestão de stake baseada na confiança do modelo
            confianca = np.mean(y_pred_proba[:, i][y_pred == i])
            stake_sugerido = min(confianca * 10, 10)  # Máximo de 10% do bankroll
            logging.info(f"Stake sugerido: {stake_sugerido:.1f}% do bankroll")
    
    return y_pred, y_pred_proba

def plot_results(model, X_test, y_test, feature_cols, window_size, le):
    """Plota visualizações detalhadas dos resultados"""
    os.makedirs('ml_model_build_xgboost/figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Matriz de confusão
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=['Derrota', 'Empate', 'Vitória'],
        yticklabels=['Derrota', 'Empate', 'Vitória']
    )
    plt.title(f'Matriz de Confusão XGBoost - Janela de {window_size} Jogos')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig(f'ml_model_build_xgboost/figures/confusion_matrix_{window_size}_{timestamp}.png')
    plt.close()
    
    # Feature Importance com valores
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Features Mais Importantes - Janela de {window_size} Jogos')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig(f'ml_model_build_xgboost/figures/feature_importance_{window_size}_{timestamp}.png')
    plt.close()
    
    # Logging das features mais importantes
    logging.info("\nTop 10 Features Mais Importantes:")
    for idx, row in feature_importance.head(10).iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

def save_model(model, scaler, le, window_size):
    """Salva o modelo, scaler e label encoder"""
    os.makedirs('ml_model_build_xgboost/ml_models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = f'ml_model_build_xgboost/ml_models/xgboost_model_{window_size}_{timestamp}.pkl'
    scaler_path = f'ml_model_build_xgboost/ml_models/xgboost_scaler_{window_size}_{timestamp}.pkl'
    le_path = f'ml_model_build_xgboost/ml_models/xgboost_label_encoder_{window_size}_{timestamp}.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    logging.info(f"Modelo, scaler e label encoder salvos em {model_path}")

def main():
    print('\n\n ---------------- START ---------------- \n')
    
    try:
        # Processar ambas as janelas de tempo
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            # Criar diretórios necessários
            os.makedirs('ml_model_build_xgboost/figures', exist_ok=True)
            os.makedirs('ml_model_build_xgboost/ml_models', exist_ok=True)
            
            # Carregar e preparar dados
            df = load_data(window_size)
            logging.info(f"Dados carregados com sucesso. Shape: {df.shape}")
            
            X_train, X_test, y_train, y_test, feature_cols, scaler, le = prepare_data(df)
            
            # Otimizar e treinar modelo
            model = optimize_hyperparameters(X_train, y_train)
            
            # Avaliar modelo
            y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, window_size, le)
            
            # Plotar resultados
            plot_results(model, X_test, y_test, feature_cols, window_size, le)
            
            # Salvar modelo
            save_model(model, scaler, le, window_size)
            
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main() 