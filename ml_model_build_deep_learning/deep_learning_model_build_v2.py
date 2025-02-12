import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import warnings
from typing import Tuple, Dict, Any, Optional
import pickle
import joblib
from imblearn.combine import SMOTEENN
import json

# Criar diretórios necessários
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = ['logs', 'figures', 'ml_models']
for dir_name in DIRS:
    dir_path = os.path.join(SCRIPT_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'logs', 'training.log')),
        logging.StreamHandler()
    ]
)

class EarlyStopping:
    """Early stopping para prevenir overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class FootballDataset(Dataset):
    """Dataset customizado para dados de futebol"""
    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("X e y devem ter o mesmo comprimento")
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DeepFootballNet(nn.Module):
    """Rede neural profunda para previsão de resultados de futebol"""
    def __init__(self, input_size: int, hidden_sizes: list = [256, 128, 64, 32]):
        super(DeepFootballNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], 3)
        
    def forward(self, x):
        x = self.feature_layers(x)
        return self.output_layer(x)

def load_data(window_size: int = 10) -> pd.DataFrame:
    """Carrega os dados do arquivo mais recente"""
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

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
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
        
        return X_train_balanced, X_test, y_train_balanced, y_test, feature_cols, scaler, le
        
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                device: torch.device,
                epochs: int = 100,
                patience: int = 7) -> Dict[str, list]:
    """Treina o modelo com early stopping e learning rate scheduling"""
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        early_stopping = EarlyStopping(patience=patience)
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Modo treino
            model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(targets).sum().item()
            
            # Modo validação
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += targets.size(0)
                    correct_val += predicted.eq(targets).sum().item()
            
            # Atualizar métricas
            train_acc = 100. * correct_train / total_train
            val_acc = 100. * correct_val / total_val
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, '
                           f'Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                           f'Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if early_stopping(val_loss):
                logging.info(f'Early stopping ativado na época {epoch+1}')
                break
        
        # Restaurar melhor modelo
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return history
        
    except Exception as e:
        logging.error(f"Erro durante o treinamento: {str(e)}")
        raise

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: torch.device,
                  le: LabelEncoder) -> Dict[str, Any]:
    """Avalia o modelo no conjunto de teste"""
    try:
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        results = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision_macro': precision_score(all_targets, all_preds, average='macro'),
            'recall_macro': recall_score(all_targets, all_preds, average='macro'),
            'f1_macro': f1_score(all_targets, all_preds, average='macro'),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'classification_report': classification_report(all_targets, all_preds)
        }
        
        # Métricas por classe
        results['metrics_by_class'] = {}
        for i, result in enumerate(['Derrota', 'Empate', 'Vitória']):
            metrics = {
                'precision': precision_score(all_targets, all_preds, labels=[i], average=None)[0],
                'recall': recall_score(all_targets, all_preds, labels=[i], average=None)[0],
                'f1': f1_score(all_targets, all_preds, labels=[i], average=None)[0]
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
        logging.error(f"Erro durante a avaliação: {str(e)}")
        raise

def plot_results(history: Dict[str, list], 
                evaluation_results: Dict[str, Any], 
                window_size: int):
    """Plota os resultados do treinamento e avaliação"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot de loss e acurácia
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'figures', f'training_history_{window_size}_{timestamp}.png'))
        plt.close()
        
        # Matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            evaluation_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Derrota', 'Empate', 'Vitória'],
            yticklabels=['Derrota', 'Empate', 'Vitória']
        )
        plt.title(f'Matriz de Confusão - Janela de {window_size} Jogos')
        plt.ylabel('Real')
        plt.xlabel('Previsto')
        
        plt.savefig(os.path.join(SCRIPT_DIR, 'figures', f'confusion_matrix_{window_size}_{timestamp}.png'))
        plt.close()
        
    except Exception as e:
        logging.error(f"Erro ao plotar resultados: {str(e)}")
        raise

def save_model(model: nn.Module, 
              scaler: StandardScaler, 
              le: LabelEncoder, 
              window_size: int,
              model_info: Optional[Dict] = None):
    """Salva o modelo e seus componentes"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salvar modelo PyTorch
        model_path = os.path.join(SCRIPT_DIR, 'ml_models', f'deep_learning_model_{window_size}_{timestamp}.pt')
        torch.save(model.state_dict(), model_path)
        
        # Salvar scaler e label encoder
        scaler_path = os.path.join(SCRIPT_DIR, 'ml_models', f'scaler_{window_size}_{timestamp}.pkl')
        le_path = os.path.join(SCRIPT_DIR, 'ml_models', f'label_encoder_{window_size}_{timestamp}.pkl')
        
        joblib.dump(scaler, scaler_path)
        joblib.dump(le, le_path)
        
        # Salvar informações adicionais
        if model_info is not None:
            # Converter arrays NumPy para listas
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            model_info_json = convert_numpy(model_info)
            info_path = os.path.join(SCRIPT_DIR, 'ml_models', f'model_info_{window_size}_{timestamp}.json')
            with open(info_path, 'w') as f:
                json.dump(model_info_json, f, indent=4)
        
        logging.info(f"Modelo e componentes salvos em {os.path.join(SCRIPT_DIR, 'ml_models')}")
        
    except Exception as e:
        logging.error(f"Erro ao salvar modelo: {str(e)}")
        raise

def main():
    """Função principal"""
    try:
        logging.info("Iniciando treinamento do modelo Deep Learning...")
        
        # Configurar device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Usando device: {device}")
        
        # Processar diferentes tamanhos de janela
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            X_train, X_test, y_train, y_test, feature_cols, scaler, le = prepare_data(df)
            
            # Criar datasets e dataloaders
            train_dataset = FootballDataset(X_train, y_train)
            test_dataset = FootballDataset(X_test, y_test)
            
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # Criar e treinar modelo
            model = DeepFootballNet(input_size=len(feature_cols)).to(device)
            history = train_model(model, train_loader, val_loader, device)
            
            # Avaliar modelo
            evaluation_results = evaluate_model(model, test_loader, device, le)
            
            # Plotar resultados
            plot_results(history, evaluation_results, window_size)
            
            # Salvar modelo e componentes
            model_info = {
                'feature_cols': feature_cols,
                'evaluation_results': evaluation_results,
                'training_history': history
            }
            save_model(model, scaler, le, window_size, model_info)
            
            # Logar resultados
            logging.info("\nResultados da Avaliação:")
            logging.info(f"Acurácia: {evaluation_results['accuracy']:.3f}")
            logging.info(f"Precision Macro: {evaluation_results['precision_macro']:.3f}")
            logging.info(f"Recall Macro: {evaluation_results['recall_macro']:.3f}")
            logging.info(f"F1 Macro: {evaluation_results['f1_macro']:.3f}")
            
            logging.info("\nMétricas por Classe:")
            for result, metrics in evaluation_results['metrics_by_class'].items():
                logging.info(f"\n{result}:")
                for metric, value in metrics.items():
                    logging.info(f"  {metric}: {value:.3f}")
                if 'retorno_esperado' in metrics:
                    logging.info(f"  Retorno Esperado: {metrics['retorno_esperado']:.2f}")
        
        logging.info("\nTreinamento concluído com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 