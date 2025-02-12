import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         classes: List[str],
                         save_path: str,
                         title: Optional[str] = None,
                         normalize: bool = True,
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plota e salva a matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        classes: Lista com nomes das classes
        save_path: Caminho para salvar o plot
        title: Título do plot (opcional)
        normalize: Se deve normalizar os valores
        figsize: Tamanho da figura
    """
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Criar plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2%' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    plt.title(title or 'Matriz de Confusão')
    plt.ylabel('Label Verdadeiro')
    plt.xlabel('Label Predito')
    
    # Salvar plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curves(y_true: np.ndarray,
                    y_pred_proba: np.ndarray,
                    classes: List[str],
                    save_path: str,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plota e salva as curvas ROC
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
        classes: Lista com nomes das classes
        save_path: Caminho para salvar o plot
        title: Título do plot (opcional)
        figsize: Tamanho da figura
    """
    
    # Criar plot
    plt.figure(figsize=figsize)
    
    # Calcular ROC para cada classe
    for i, class_name in enumerate(classes):
        # Converter para one-vs-rest
        y_true_binary = (y_true == i).astype(int)
        y_pred_class = y_pred_proba[:, i]
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_class)
        roc_auc = auc(fpr, tpr)
        
        # Plotar curva
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(title or 'Curvas ROC')
    plt.legend(loc="lower right")
    
    # Salvar plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance(feature_importance: np.ndarray,
                          feature_names: List[str],
                          save_path: str,
                          title: str = 'Importância das Features',
                          figsize: Tuple[float, float] = (6.4, 4.8)) -> None:
    """
    Plota importância das features
    
    Args:
        feature_importance: Array com importâncias
        feature_names: Lista com nomes das features
        save_path: Caminho para salvar
        title: Título do gráfico
        figsize: Tamanho da figura
    """
    # Criar figura
    plt.figure(figsize=figsize)
    
    # Ordenar importâncias
    indices = np.argsort(feature_importance)[::-1]
    
    # Criar barplot
    plt.bar(range(len(indices)), feature_importance[indices])
    
    # Configurar eixos
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importância')
    
    # Adicionar título
    plt.title(title)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(save_path)
    plt.close()

def plot_prediction_distribution(predictions: Dict[str, float],
                               save_path: str,
                               title: str = 'Distribuição das Predições',
                               figsize: Tuple[float, float] = (6.4, 4.8)) -> None:
    """
    Plota distribuição das predições
    
    Args:
        predictions: Dicionário com probabilidades
        save_path: Caminho para salvar
        title: Título do gráfico
        figsize: Tamanho da figura
    """
    # Criar figura
    plt.figure(figsize=figsize)
    
    # Criar barplot
    plt.bar(predictions.keys(), predictions.values())
    
    # Adicionar valores
    for i, (_, value) in enumerate(predictions.items()):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')
    
    # Configurar eixos
    plt.xlabel('Resultados')
    plt.ylabel('Probabilidade')
    
    # Adicionar título
    plt.title(title)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(prob_true: np.ndarray,
                          prob_pred: np.ndarray,
                          save_path: str,
                          title: str = 'Curva de Calibração',
                          figsize: Tuple[float, float] = (6.4, 4.8)) -> None:
    """
    Plota curva de calibração
    
    Args:
        prob_true: Probabilidades reais
        prob_pred: Probabilidades preditas
        save_path: Caminho para salvar
        title: Título do gráfico
        figsize: Tamanho da figura
    """
    # Criar figura
    plt.figure(figsize=figsize)
    
    # Plotar linha diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Perfeita Calibração')
    
    # Plotar curva de calibração
    plt.plot(prob_pred, prob_true, 'o-', label='Modelo')
    
    # Configurar eixos
    plt.xlabel('Probabilidade Predita')
    plt.ylabel('Probabilidade Real')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Adicionar título e legenda
    plt.title(title)
    plt.legend()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(save_path)
    plt.close()

def plot_metrics_over_time(metrics_df: pd.DataFrame,
                          save_path: str,
                          title: str = 'Métricas ao Longo do Tempo',
                          figsize: Tuple[float, float] = (6.4, 4.8)) -> None:
    """
    Plota métricas ao longo do tempo
    
    Args:
        metrics_df: DataFrame com métricas
        save_path: Caminho para salvar
        title: Título do gráfico
        figsize: Tamanho da figura
    """
    # Criar figura
    plt.figure(figsize=figsize)
    
    # Plotar métricas
    for col in metrics_df.columns:
        if col != 'date':
            plt.plot(metrics_df['date'], metrics_df[col], label=col)
    
    # Configurar eixos
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.xticks(rotation=45)
    
    # Adicionar título e legenda
    plt.title(title)
    plt.legend()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(models_metrics: Dict[str, Dict[str, float]],
                         metric_name: str,
                         save_path: str,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plota e salva comparação de métricas entre modelos
    
    Args:
        models_metrics: Dicionário com métricas dos modelos
        metric_name: Nome da métrica a ser comparada
        save_path: Caminho para salvar o plot
        title: Título do plot (opcional)
        figsize: Tamanho da figura
    """
    
    # Preparar dados
    models = list(models_metrics.keys())
    metrics = [metrics[metric_name] for metrics in models_metrics.values()]
    
    # Criar plot
    plt.figure(figsize=figsize)
    plt.bar(models, metrics)
    plt.title(title or f'Comparação de {metric_name} entre Modelos')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    
    # Adicionar valores
    for i, metric in enumerate(metrics):
        plt.text(i, metric, f'{metric:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Salvar plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_with_custom_style(figsize: Tuple[float, float] = (6.4, 4.8)) -> None:
    """
    Configura estilo personalizado para plots
    
    Args:
        figsize: Tamanho da figura
    """
    # Configurar estilo
    plt.style.use('default')
    
    # Configurar parâmetros
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16 