# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:57:52 2020
@author: mhayt
"""

print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import xgboost as xgb
from sklearn.svm import SVC
import joblib
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#----------------------------- LOAD DATA -------------------------

def load_data(window_size=10):
    try:
        # Encontrar o arquivo mais recente
        df_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') if f.startswith('prem_df_for_ml_')]
        if not df_files:
            raise FileNotFoundError("No dataframe files found")
        
        # Procurar arquivo específico para o tamanho da janela
        df_file = f'prem_df_for_ml_{window_size}_v2.txt'
        if df_file not in df_files:
            raise FileNotFoundError(f"No dataframe file found for window size {window_size}")
        
        with open(f'prem_clean_fixtures_and_dataframes/{df_file}', 'rb') as myFile:
            df = pickle.load(myFile)
            
        logging.info(f"Successfully loaded {window_size}-game window dataframe")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

#---------------------------- RANDOM FOREST BUILD ---------------------------

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list, StandardScaler, LabelEncoder]:
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
        
        return X, y, feature_cols, scaler, le
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def create_ensemble():
    """Cria um ensemble de modelos"""
    # Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    )
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    
    # Criar ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_model),
            ('svm', svm)
        ],
        voting='soft'
    )
    
    return ensemble

def optimize_hyperparameters(X_train, y_train):
    """Otimiza hiperparâmetros usando GridSearchCV com técnicas avançadas"""
    # Random Forest base
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # XGBoost base
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    )
    
    # Criar ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    
    # Parâmetros para otimização
    param_grid = {
        'rf__max_depth': [20, 30, None],
        'rf__min_samples_split': [2, 5],
        'rf__max_features': ['sqrt', 'log2'],
        
        'xgb__subsample': [0.8, 0.9],
        'xgb__colsample_bytree': [0.8, 0.9]
    }
    
    # Grid Search com validação cruzada estratificada
    grid_search = GridSearchCV(
        estimator=ensemble,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        scoring='f1_weighted',
        verbose=1
    )
    
    # SMOTE para balanceamento
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, window_size):
    """Avalia o modelo com métricas específicas para apostas"""
    # Fazer previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    logging.info(f"\nResultados para janela de {window_size} jogos:")
    logging.info(f"Acurácia: {accuracy:.3f}")
    logging.info(f"\nRelatório de classificação:\n{class_report}")
    
    # Métricas específicas para apostas
    for i, result in enumerate(['Derrota', 'Empate', 'Vitória']):
        precision = precision_score(y_test, y_pred, labels=[i], average=None)[0]
        recall = recall_score(y_test, y_pred, labels=[i], average=None)[0]
        f1 = f1_score(y_test, y_pred, labels=[i], average=None)[0]
        
        logging.info(f"\nMétricas para {result}:")
        logging.info(f"Precisão: {precision:.3f}")
        logging.info(f"Recall: {recall:.3f}")
        logging.info(f"F1-Score: {f1:.3f}")
        
        # Calcular retorno potencial baseado em odds médias
        if precision > 0.6:  # Apenas mostrar oportunidades com alta precisão
            if result == 'Vitória':
                odds_media = 2.5
            elif result == 'Empate':
                odds_media = 3.5
            else:
                odds_media = 4.0
                
            retorno_esperado = (precision * odds_media) - 1
            logging.info(f"Retorno esperado com odds {odds_media}: {retorno_esperado:.2f}")
    
    # Plotar curvas de precisão-recall
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(['Derrota', 'Empate', 'Vitória']):
        precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba[:, i])
        avg_precision = average_precision_score(y_test == i, y_pred_proba[:, i])
        plt.plot(recall, precision, label=f'{result} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title(f'Curvas de Precisão-Recall (Janela de {window_size} jogos)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/precision_recall_{window_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()
    
    return model

def plot_results(evaluation_results, window_size):
    """Plota resultados detalhados com foco em métricas para apostas"""
    os.makedirs('ml_model_build_random_forest/figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Confusion Matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(
        evaluation_results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Derrota', 'Empate', 'Vitória'],
        yticklabels=['Derrota', 'Empate', 'Vitória']
    )
    plt.title(f'Matriz de Confusão - Janela de {window_size} Jogos')
    plt.ylabel('Valor Real')
    plt.xlabel('Previsão')
    plt.savefig(f'ml_model_build_random_forest/figures/confusion_matrix_{window_size}_{timestamp}.png')
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10,7))
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
    plt.title(f'Curvas ROC - Janela de {window_size} Jogos')
    plt.legend(loc='lower right')
    plt.savefig(f'ml_model_build_random_forest/figures/roc_curves_{window_size}_{timestamp}.png')
    plt.close()
    
    # Precision-Recall Curves
    plt.figure(figsize=(10,7))
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(
            evaluation_results['pr_data'][1][i],
            evaluation_results['pr_data'][0][i],
            color=color,
            label=f'{label} (AP = {evaluation_results["pr_data"][2][i]:.2f})'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curvas Precision-Recall - Janela de {window_size} Jogos')
    plt.legend(loc='lower left')
    plt.savefig(f'ml_model_build_random_forest/figures/pr_curves_{window_size}_{timestamp}.png')
    plt.close()

def save_model(model, window_size):
    """Salva o modelo treinado"""
    os.makedirs('ml_model_build_random_forest/ml_models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'ml_model_build_random_forest/ml_models/random_forest_model_{window_size}_{timestamp}.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved to {model_path}")

def plot_feature_importance(model, feature_names, window_size):
    """Plota a importância das features do modelo Random Forest"""
    # Definir diretório raiz do projeto
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Criar diretório para figuras se não existir
    os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest', 'figures'), exist_ok=True)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info(f"\nTop 10 Features Mais Importantes (Janela de {window_size} jogos):")
    logging.info(feature_importance.head(10))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'15 Features Mais Importantes - Janela de {window_size} Jogos')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Salvar figura usando o caminho completo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest', 'figures', 
                           f'feature_importance_{window_size}_{timestamp}.png')
    plt.savefig(fig_path)
    plt.close()
    
    logging.info(f"Gráfico de importância das features salvo em: {fig_path}")

def main():
    """Função principal para treinamento e avaliação do modelo"""
    logging.basicConfig(level=logging.INFO)
    
    # Definir diretório raiz do projeto
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Criar diretórios necessários
    os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest', 'ml_models'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest', 'figures'), exist_ok=True)
    
    # Carregar e preparar dados para janela de 5 jogos
    logging.info("\nProcessando janela de 5 jogos...")
    data_file = os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', 'prem_df_for_ml_5_v2.txt')
    with open(data_file, 'rb') as myFile:
        df_5 = pickle.load(myFile)
    
    X_5, y_5, feature_cols_5, scaler_5, le_5 = prepare_data(df_5)
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(
        X_5, y_5, test_size=0.2, random_state=42, stratify=y_5)
    
    # Treinar e otimizar modelo para 5 jogos
    model_5 = optimize_hyperparameters(X_train_5, y_train_5)
    evaluate_model(model_5, X_test_5, y_test_5, window_size=5)
    
    # Salvar modelo de 5 jogos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_5 = os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest', 'ml_models', f'random_forest_model_5_{timestamp}.pkl')
    joblib.dump(model_5, model_path_5)
    logging.info(f"Modelo de 5 jogos salvo em: {model_path_5}")
    
    # Carregar e preparar dados para janela de 10 jogos
    logging.info("\nProcessando janela de 10 jogos...")
    data_file_10 = os.path.join(PROJECT_ROOT, 'prem_clean_fixtures_and_dataframes', 'prem_df_for_ml_10_v2.txt')
    with open(data_file_10, 'rb') as myFile:
        df_10 = pickle.load(myFile)
    
    X_10, y_10, feature_cols_10, scaler_10, le_10 = prepare_data(df_10)
    X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(
        X_10, y_10, test_size=0.2, random_state=42, stratify=y_10
    )
    
    # Treinar e otimizar modelo para 10 jogos
    model_10 = optimize_hyperparameters(X_train_10, y_train_10)
    evaluate_model(model_10, X_test_10, y_test_10, window_size=10)
    
    # Salvar modelo de 10 jogos
    model_path_10 = os.path.join(PROJECT_ROOT, 'ml_model_build_random_forest', 'ml_models', f'random_forest_model_10_{timestamp}.pkl')
    joblib.dump(model_10, model_path_10)
    logging.info(f"Modelo de 10 jogos salvo em: {model_path_10}")
    
    # Plotar importância das features para ambos os modelos
    if hasattr(model_5, 'named_estimators_'):
        rf_model_5 = model_5.named_estimators_['rf']
        plot_feature_importance(rf_model_5, feature_cols_5, window_size=5)
        
    if hasattr(model_10, 'named_estimators_'):
        rf_model_10 = model_10.named_estimators_['rf']
        plot_feature_importance(rf_model_10, feature_cols_10, window_size=10)
    
    logging.info("\nTreinamento e avaliação concluídos com sucesso!")

if __name__ == "__main__":
    main()
    print('\n ----------------- END ----------------- \n')
