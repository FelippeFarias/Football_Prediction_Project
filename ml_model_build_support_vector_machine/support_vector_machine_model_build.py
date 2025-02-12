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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#----------------------------- LOAD DATA -------------------------

def load_data(window_size=10):
    try:
        # Encontrar o arquivo mais recente
        df_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') if f.endswith(f'_v2.txt')]
        if not df_files:
            raise FileNotFoundError("No dataframe files found")
        
        df_file = max([f for f in df_files if f'{window_size}_v2.txt' in f], 
                     key=lambda x: os.path.getctime(os.path.join('prem_clean_fixtures_and_dataframes', x)))
        
        with open(f'prem_clean_fixtures_and_dataframes/{df_file}', 'rb') as myFile:
            df = pickle.load(myFile)
            
        logging.info(f"Successfully loaded {window_size}-game window dataframe")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

#---------------------------- SVM BUILD ---------------------------

def prepare_data(df):
    """Prepara os dados para treinamento com normalização e SMOTE"""
    # Removendo colunas que não serão usadas para treinamento
    feature_cols = [col for col in df.columns if col not in ['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']]
    X = df[feature_cols]
    y = df['Team Result Indicator']
    
    # Divisão treino/teste estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Converter para DataFrame mantendo nomes das colunas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def optimize_hyperparameters(X_train, y_train):
    """Otimiza hiperparâmetros usando GridSearchCV"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced', None],
        'degree': [2, 3] # para kernel poly
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
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

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Avalia o modelo com métricas detalhadas"""
    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    
    # ROC Curve para cada classe
    n_classes = len(np.unique(y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'cv_scores': cv_scores,
        'roc_data': (fpr, tpr, roc_auc)
    }

def plot_results(evaluation_results, window_size):
    """Plota resultados detalhados da avaliação"""
    # Criar diretório para figuras
    os.makedirs('ml_model_build_support_vector_machine/figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Confusion Matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(evaluation_results['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['Derrota', 'Empate', 'Vitória'],
                yticklabels=['Derrota', 'Empate', 'Vitória'])
    plt.title(f'Matriz de Confusão SVM - Janela de {window_size} Jogos')
    plt.ylabel('Valor Real')
    plt.xlabel('Previsão')
    plt.savefig(f'ml_model_build_support_vector_machine/figures/confusion_matrix_{window_size}_{timestamp}.png')
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10,7))
    colors = ['blue', 'red', 'green']
    labels = ['Derrota', 'Empate', 'Vitória']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(evaluation_results['roc_data'][0][i], 
                evaluation_results['roc_data'][1][i], 
                color=color,
                label=f'{label} (AUC = {evaluation_results["roc_data"][2][i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curvas ROC SVM - Janela de {window_size} Jogos')
    plt.legend(loc='lower right')
    plt.savefig(f'ml_model_build_support_vector_machine/figures/roc_curves_{window_size}_{timestamp}.png')
    plt.close()

def save_model_and_scaler(model, scaler, window_size):
    """Salva o modelo treinado e o scaler"""
    os.makedirs('ml_model_build_support_vector_machine/ml_models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Salvar modelo
    model_path = f'ml_model_build_support_vector_machine/ml_models/svm_model_{window_size}_{timestamp}.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved to {model_path}")
    
    # Salvar scaler
    scaler_path = f'ml_model_build_support_vector_machine/ml_models/svm_scaler_{window_size}_{timestamp}.pkl'
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    logging.info(f"Scaler saved to {scaler_path}")

def main():
    try:
        # Processar ambas as janelas de tempo
        for window_size in [5, 10]:
            logging.info(f"\nProcessing {window_size}-game window")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_data(df)
            
            # Otimizar e treinar modelo
            logging.info("Optimizing hyperparameters...")
            model = optimize_hyperparameters(X_train, y_train)
            
            # Avaliar modelo
            logging.info("Evaluating model...")
            evaluation_results = evaluate_model(model, X_test, y_test, X_train, y_train)
            
            # Logging dos resultados
            logging.info(f"\nClassification Report:\n{evaluation_results['classification_report']}")
            logging.info(f"Cross-validation scores: {evaluation_results['cv_scores'].mean():.3f} (+/- {evaluation_results['cv_scores'].std() * 2:.3f})")
            
            # Plotar resultados
            plot_results(evaluation_results, window_size)
            
            # Salvar modelo e scaler
            save_model_and_scaler(model, scaler, window_size)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    print('\n ----------------- END ----------------- \n')
