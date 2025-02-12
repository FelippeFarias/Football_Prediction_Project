import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import pickle
from typing import Dict, Tuple, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_evaluation/model_evaluation.log'),
        logging.StreamHandler()
    ]
)

class ModelEvaluator:
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.2, random_state: int = 42):
        """
        Inicializa o avaliador de modelos.
        
        Args:
            test_size: Proporção do conjunto de teste
            validation_size: Proporção do conjunto de validação
            random_state: Semente aleatória para reprodutibilidade
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Divide os dados em conjuntos de treino, validação e teste.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple contendo X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Primeiro split para separar o conjunto de teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Segundo split para separar treino e validação
        val_size = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        logging.info(f"Shapes dos conjuntos:")
        logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        logging.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Avalia as previsões do modelo usando múltiplas métricas.
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            y_pred_proba: Probabilidades previstas (opcional)
            
        Returns:
            Dicionário com as métricas calculadas
        """
        results = {}
        
        # Métricas básicas
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        results['classification_report'] = classification_report(y_true, y_pred)
        
        # Métricas por classe
        results['metrics_by_class'] = {}
        for i, result in enumerate(['Derrota', 'Empate', 'Vitória']):
            class_metrics = {
                'precision': precision_score(y_true, y_pred, labels=[i], average=None)[0],
                'recall': recall_score(y_true, y_pred, labels=[i], average=None)[0],
                'f1': f1_score(y_true, y_pred, labels=[i], average=None)[0]
            }
            
            # Calcular retorno potencial baseado em odds médias
            if class_metrics['precision'] > 0.6:
                odds_media = {
                    'Vitória': 2.5,
                    'Empate': 3.5,
                    'Derrota': 4.0
                }[result]
                
                retorno_esperado = (class_metrics['precision'] * odds_media) - 1
                class_metrics['retorno_esperado'] = retorno_esperado
            
            results['metrics_by_class'][result] = class_metrics
        
        return results
    
    def plot_evaluation_results(self, results: Dict[str, Any], window_size: int):
        """
        Plota os resultados da avaliação.
        
        Args:
            results: Dicionário com os resultados da avaliação
            window_size: Tamanho da janela de jogos
        """
        os.makedirs('ml_model_evaluation/figures', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Matriz de Confusão
        plt.figure(figsize=(10,7))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Derrota', 'Empate', 'Vitória'],
            yticklabels=['Derrota', 'Empate', 'Vitória']
        )
        plt.title(f'Matriz de Confusão - Janela de {window_size} Jogos')
        plt.ylabel('Valor Real')
        plt.xlabel('Previsão')
        plt.savefig(f'ml_model_evaluation/figures/confusion_matrix_{window_size}_{timestamp}.png')
        plt.close()
        
        # Gráfico de barras com métricas por classe
        plt.figure(figsize=(12,6))
        metrics = ['precision', 'recall', 'f1']
        x = np.arange(len(results['metrics_by_class']))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results['metrics_by_class'][result][metric] 
                     for result in results['metrics_by_class']]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Classes')
        plt.ylabel('Valor')
        plt.title(f'Métricas por Classe - Janela de {window_size} Jogos')
        plt.xticks(x + width, results['metrics_by_class'].keys())
        plt.legend()
        plt.savefig(f'ml_model_evaluation/figures/metrics_by_class_{window_size}_{timestamp}.png')
        plt.close()
    
    def save_results(self, results: Dict[str, Any], window_size: int):
        """
        Salva os resultados da avaliação em um arquivo.
        
        Args:
            results: Dicionário com os resultados da avaliação
            window_size: Tamanho da janela de jogos
        """
        os.makedirs('ml_model_evaluation/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salvar resultados em formato pickle
        with open(f'ml_model_evaluation/results/evaluation_{window_size}_{timestamp}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Salvar um resumo em formato texto
        with open(f'ml_model_evaluation/results/summary_{window_size}_{timestamp}.txt', 'w') as f:
            f.write(f"Avaliação do Modelo - Janela de {window_size} Jogos\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelo utilizado: {results.get('model_used', 'N/A')}\n\n")
            
            # Resultados da validação
            f.write("=== Resultados da Validação ===\n")
            val_results = results.get('validation', {})
            f.write(f"Acurácia Global: {val_results.get('accuracy', 0):.3f}\n\n")
            f.write("Relatório de Classificação:\n")
            f.write(str(val_results.get('classification_report', 'N/A')))
            f.write("\n\nMétricas por Classe:\n")
            
            for result, metrics in val_results.get('metrics_by_class', {}).items():
                f.write(f"\n{result}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.3f}\n")
                if 'retorno_esperado' in metrics:
                    f.write(f"  Retorno Esperado: {metrics['retorno_esperado']:.2f}\n")
            
            # Resultados do teste
            f.write("\n\n=== Resultados do Teste ===\n")
            test_results = results.get('test', {})
            f.write(f"Acurácia Global: {test_results.get('accuracy', 0):.3f}\n\n")
            f.write("Relatório de Classificação:\n")
            f.write(str(test_results.get('classification_report', 'N/A')))
            f.write("\n\nMétricas por Classe:\n")
            
            for result, metrics in test_results.get('metrics_by_class', {}).items():
                f.write(f"\n{result}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.3f}\n")
                if 'retorno_esperado' in metrics:
                    f.write(f"  Retorno Esperado: {metrics['retorno_esperado']:.2f}\n") 