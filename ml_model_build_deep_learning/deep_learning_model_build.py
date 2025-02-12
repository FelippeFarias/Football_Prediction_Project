import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import warnings
from typing import Tuple, Dict, Any, Optional
import pickle
import json

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_build_deep_learning/training.log'),
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
        
        if not hidden_sizes or not all(isinstance(x, int) and x > 0 for x in hidden_sizes):
            raise ValueError("hidden_sizes deve ser uma lista de inteiros positivos")
        
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

def validate_data(df: pd.DataFrame) -> None:
    """Valida os dados de entrada"""
    if df is None or df.empty:
        raise ValueError("DataFrame está vazio")
    
    required_columns = ['Team Result Indicator', 'Fixture ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
    
    if df['Team Result Indicator'].isna().any():
        raise ValueError("Existem valores ausentes na coluna 'Team Result Indicator'")

def load_data(window_size: int = 10) -> pd.DataFrame:
    """Carrega os dados do arquivo mais recente"""
    try:
        base_path = 'prem_clean_fixtures_and_dataframes'
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Diretório {base_path} não encontrado")
        
        df_files = [f for f in os.listdir(base_path) if f.endswith(f'_v2.txt')]
        if not df_files:
            raise FileNotFoundError("Nenhum arquivo de dados encontrado")
        
        df_file = max([f for f in df_files if f'{window_size}_v2.txt' in f], 
                     key=lambda x: os.path.getctime(os.path.join(base_path, x)))
        
        file_path = os.path.join(base_path, df_file)
        with open(file_path, 'rb') as myFile:
            df = pickle.load(myFile)
        
        validate_data(df)
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

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                device: torch.device,
                epochs: int = 100,
                patience: int = 7) -> Dict[str, list]:
    """Treina o modelo com monitoramento de métricas e early stopping"""
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
            logging.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if early_stopping(val_loss):
            logging.info(f'Early stopping ativado na época {epoch+1}')
            break
    
    # Restaurar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: torch.device,
                  le: LabelEncoder) -> Dict[str, Any]:
    """Avalia o modelo com métricas detalhadas"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Converter para labels originais
    all_preds_original = le.inverse_transform(all_preds)
    all_targets_original = le.inverse_transform(all_targets)
    
    # Calcular métricas
    report = classification_report(all_targets_original, all_preds_original)
    cm = confusion_matrix(all_targets_original, all_preds_original)
    
    return {
        'predictions': all_preds_original,
        'targets': all_targets_original,
        'probabilities': all_probs,
        'report': report,
        'confusion_matrix': cm
    }

def plot_results(history: Dict[str, list], 
                evaluation_results: Dict[str, Any], 
                window_size: int):
    """Plota visualizações dos resultados"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('ml_model_build_deep_learning/figures', exist_ok=True)
    
    # Plot do histórico de treinamento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'ml_model_build_deep_learning/figures/training_history_{window_size}_{timestamp}.png')
    plt.close()
    
    # Matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(evaluation_results['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['Derrota', 'Empate', 'Vitória'],
                yticklabels=['Derrota', 'Empate', 'Vitória'])
    plt.title(f'Matriz de Confusão - Deep Learning - Janela de {window_size} Jogos')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig(f'ml_model_build_deep_learning/figures/confusion_matrix_{window_size}_{timestamp}.png')
    plt.close()

def save_model(model: nn.Module, 
              scaler: StandardScaler, 
              le: LabelEncoder, 
              window_size: int,
              model_info: Optional[Dict] = None):
    """Salva o modelo e os transformadores com informações adicionais"""
    try:
        base_path = 'ml_model_build_deep_learning/ml_models'
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_path = os.path.join(base_path, f'deep_learning_{window_size}_{timestamp}.pt')
        scaler_path = os.path.join(base_path, f'scaler_{window_size}_{timestamp}.pkl')
        le_path = os.path.join(base_path, f'label_encoder_{window_size}_{timestamp}.pkl')
        info_path = os.path.join(base_path, f'model_info_{window_size}_{timestamp}.json')
        
        # Salvar modelo
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'input_size': model.feature_layers[0].in_features,
            'timestamp': timestamp
        }, model_path)
        
        # Salvar transformadores
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)
            
        # Salvar informações adicionais
        if model_info:
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
        
        logging.info(f"Modelo e transformadores salvos em {base_path}")
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'le_path': le_path,
            'info_path': info_path if model_info else None
        }
    except Exception as e:
        logging.error(f"Erro ao salvar modelo: {str(e)}")
        raise

def main():
    print('\n\n ---------------- START ---------------- \n')
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Usando dispositivo: {device}")
        
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
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # Criar e treinar modelo
            model = DeepFootballNet(input_size=len(feature_cols)).to(device)
            history = train_model(model, train_loader, val_loader, device)
            
            # Avaliar modelo
            evaluation_results = evaluate_model(model, test_loader, device, le)
            logging.info(f"\nRelatório de Classificação:\n{evaluation_results['report']}")
            
            # Plotar resultados
            plot_results(history, evaluation_results, window_size)
            
            # Salvar modelo com informações adicionais
            model_info = {
                'window_size': window_size,
                'feature_columns': feature_cols,
                'training_history': history,
                'evaluation_metrics': {
                    'classification_report': evaluation_results['report'],
                    'confusion_matrix': evaluation_results['confusion_matrix'].tolist()
                },
                'model_parameters': {
                    'input_size': len(feature_cols),
                    'hidden_sizes': [256, 128, 64, 32],
                    'dropout_rate': 0.3
                }
            }
            
            save_model(model, scaler, le, window_size, model_info)
            
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main() 