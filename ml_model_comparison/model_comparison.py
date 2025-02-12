import pandas as pd
import numpy as np
import pickle
import os
import logging
import joblib
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_comparison/comparison.log'),
        logging.StreamHandler()
    ]
)

class ModelComparison:
    def __init__(self):
        """Inicializa o comparador de modelos"""
        self.models = {}
        self.results = {}
        self.best_models = {}
        
        # Criar diretórios necessários
        os.makedirs('ml_model_comparison/results', exist_ok=True)
        os.makedirs('ml_model_comparison/figures', exist_ok=True)
        os.makedirs('ml_model_comparison/models', exist_ok=True)
        
    def load_data(self, window_size: int = 10) -> pd.DataFrame:
        """Carrega os dados para treinamento"""
        try:
            df_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') 
                       if f.startswith(f'prem_df_for_ml_{window_size}_')]
            
            if not df_files:
                raise FileNotFoundError(f"Nenhum arquivo encontrado para janela de {window_size} jogos")
            
            latest_file = max(df_files, key=lambda x: os.path.getctime(
                os.path.join('prem_clean_fixtures_and_dataframes', x)))
            
            with open(f'prem_clean_fixtures_and_dataframes/{latest_file}', 'rb') as f:
                df = pickle.load(f)
                
            logging.info(f"Dados carregados: {latest_file}")
            return df
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """Prepara os dados para treinamento"""
        try:
            # Remover colunas não necessárias
            feature_cols = [col for col in df.columns 
                          if col not in ['Fixture ID', 'Result Indicator', 'Team', 'Game Date']]
            
            # Preparar features e target
            X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
            y = df['Result Indicator'].values
            
            # Preencher valores ausentes
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
            
            return X_train_balanced, X_test, y_train_balanced, y_test, feature_cols
            
        except Exception as e:
            logging.error(f"Erro ao preparar dados: {str(e)}")
            raise
            
    def initialize_models(self):
        """Inicializa todos os modelos disponíveis"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
    def evaluate_model(self, name: str, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Avalia um modelo específico"""
        try:
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calcular métricas
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'model': model
            }
            
            # Métricas por classe
            results['metrics_by_class'] = {}
            for i, result in enumerate(['Derrota', 'Empate', 'Vitória']):
                metrics = {
                    'precision': precision_score(y_test, y_pred, labels=[i], average=None)[0],
                    'recall': recall_score(y_test, y_pred, labels=[i], average=None)[0],
                    'f1': f1_score(y_test, y_pred, labels=[i], average=None)[0]
                }
                
                # Calcular retorno esperado
                if metrics['precision'] > 0.6:
                    odds_media = {
                        'Vitória': 2.5,
                        'Empate': 3.5,
                        'Derrota': 4.0
                    }[result]
                    metrics['retorno_esperado'] = (metrics['precision'] * odds_media) - 1
                    
                results['metrics_by_class'][result] = metrics
            
            return results
            
        except Exception as e:
            logging.error(f"Erro ao avaliar modelo {name}: {str(e)}")
            raise
            
    def compare_models(self, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray):
        """Compara todos os modelos"""
        try:
            for name, model in self.models.items():
                logging.info(f"\nAvaliando modelo: {name}")
                self.results[name] = self.evaluate_model(
                    name, model, X_train, X_test, y_train, y_test
                )
                
            # Identificar melhores modelos
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            for metric in metrics:
                best_model = max(self.results.items(), 
                               key=lambda x: x[1][metric])
                self.best_models[metric] = best_model[0]
                
            logging.info("\nMelhores modelos por métrica:")
            for metric, model in self.best_models.items():
                logging.info(f"{metric}: {model} ({self.results[model][metric]:.3f})")
                
        except Exception as e:
            logging.error(f"Erro na comparação de modelos: {str(e)}")
            raise
            
    def plot_results(self):
        """Plota os resultados da comparação"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Comparação de métricas
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            model_names = list(self.results.keys())
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(model_names))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                values = [self.results[model][metric] for model in model_names]
                plt.bar(x + i*width, values, width, label=metric)
            
            plt.xlabel('Modelos')
            plt.ylabel('Score')
            plt.title('Comparação de Performance dos Modelos')
            plt.xticks(x + width*1.5, model_names, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f'ml_model_comparison/figures/model_comparison_{timestamp}.png')
            plt.close()
            
            # Matrizes de confusão
            for name, results in self.results.items():
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    results['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Derrota', 'Empate', 'Vitória'],
                    yticklabels=['Derrota', 'Empate', 'Vitória']
                )
                plt.title(f'Matriz de Confusão - {name}')
                plt.ylabel('Real')
                plt.xlabel('Previsto')
                
                plt.savefig(f'ml_model_comparison/figures/confusion_matrix_{name}_{timestamp}.png')
                plt.close()
                
        except Exception as e:
            logging.error(f"Erro ao plotar resultados: {str(e)}")
            raise
            
    def save_results(self):
        """Salva os resultados da comparação"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Salvar resultados em formato pickle
            with open(f'ml_model_comparison/results/comparison_results_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.results, f)
            
            # Salvar resumo em texto
            with open(f'ml_model_comparison/results/summary_{timestamp}.txt', 'w') as f:
                f.write("Comparação de Modelos de Machine Learning\n")
                f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for name, results in self.results.items():
                    f.write(f"\n=== {name.upper()} ===\n")
                    f.write(f"Acurácia: {results['accuracy']:.3f}\n")
                    f.write(f"Precision Macro: {results['precision_macro']:.3f}\n")
                    f.write(f"Recall Macro: {results['recall_macro']:.3f}\n")
                    f.write(f"F1 Macro: {results['f1_macro']:.3f}\n\n")
                    
                    f.write("Relatório de Classificação:\n")
                    f.write(results['classification_report'])
                    f.write("\n")
                    
                    f.write("Métricas por Classe:\n")
                    for result, metrics in results['metrics_by_class'].items():
                        f.write(f"\n{result}:\n")
                        for metric, value in metrics.items():
                            f.write(f"  {metric}: {value:.3f}\n")
                        if 'retorno_esperado' in metrics:
                            f.write(f"  Retorno Esperado: {metrics['retorno_esperado']:.2f}\n")
                            
                f.write("\n=== MELHORES MODELOS ===\n")
                for metric, model in self.best_models.items():
                    f.write(f"{metric}: {model} ({self.results[model][metric]:.3f})\n")
                    
            logging.info(f"Resultados salvos em ml_model_comparison/results/")
            
        except Exception as e:
            logging.error(f"Erro ao salvar resultados: {str(e)}")
            raise
            
def main():
    """Função principal"""
    try:
        logging.info("Iniciando comparação de modelos...")
        
        # Inicializar comparador
        comparator = ModelComparison()
        
        # Carregar e preparar dados
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            df = comparator.load_data(window_size)
            X_train, X_test, y_train, y_test, feature_cols = comparator.prepare_data(df)
            
            # Inicializar modelos
            comparator.initialize_models()
            
            # Comparar modelos
            comparator.compare_models(X_train, X_test, y_train, y_test)
            
            # Plotar e salvar resultados
            comparator.plot_results()
            comparator.save_results()
            
        logging.info("\nComparação de modelos concluída!")
        
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 