import pandas as pd
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import warnings

# Configurar warnings
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
        y_encoded = le.fit_transform(y)
        y = keras.utils.to_categorical(y_encoded)
        
        # Divisão treino/teste estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        logging.info(f"Shape após split: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Balanceamento com SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train.argmax(axis=1))
        y_train_balanced = keras.utils.to_categorical(y_train_balanced)
        
        logging.info(f"Shape após balanceamento: X_train={X_train_balanced.shape}")
        
        # Verificar valores ausentes
        if np.isnan(X_train_balanced).any() or np.isnan(X_test_scaled).any():
            logging.warning("Detectados valores ausentes nos dados!")
            
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test, feature_cols, scaler, le
        
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def create_model(input_shape, learning_rate=0.001):
    """Cria o modelo de rede neural com arquitetura otimizada"""
    try:
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            
            # Primeira camada densa com regularização
            layers.Dense(256, 
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Segunda camada densa
            layers.Dense(128, 
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Terceira camada densa
            layers.Dense(64, 
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Quarta camada densa
            layers.Dense(32, 
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            # Camada de saída
            layers.Dense(3, activation='softmax')
        ])
        
        # Compilar modelo com otimizador personalizado
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
        
    except Exception as e:
        logging.error(f"Erro ao criar modelo: {str(e)}")
        raise

def train_model(model, X_train, y_train, X_test, y_test, window_size):
    """Treina o modelo com callbacks avançados e early stopping"""
    try:
        # Criar diretório para checkpoints
        checkpoint_dir = f'ml_model_build_neural_network/checkpoints/window_{window_size}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callbacks avançados
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_accuracy:.3f}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=f'ml_model_build_neural_network/logs/window_{window_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        # Calcular pesos das classes
        y_integers = np.argmax(y_train, axis=1)
        class_weights = dict(zip(
            np.unique(y_integers),
            1 / np.bincount(y_integers) * len(y_integers) / 3
        ))
        
        # Treinar modelo
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
            verbose=1,
            class_weight=class_weights
        )
        
        return model, history
        
    except Exception as e:
        logging.error(f"Erro ao treinar modelo: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, window_size, le):
    """Avalia o modelo com métricas específicas para apostas"""
    try:
        # Previsões
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Converter para labels originais
        y_test_original = le.inverse_transform(y_test_classes)
        y_pred_original = le.inverse_transform(y_pred)
        
        # Métricas básicas
        logging.info(f"\nResultados para janela de {window_size} jogos:")
        logging.info(f"\nRelatório de classificação:\n{classification_report(y_test_original, y_pred_original)}")
        
        # Métricas específicas para apostas
        resultados = ['Derrota', 'Empate', 'Vitória']
        odds_medias = {'Derrota': 4.0, 'Empate': 3.5, 'Vitória': 2.5}
        
        for i, resultado in enumerate(resultados):
            precision = precision_recall_curve(y_test_classes == i, y_pred_proba[:, i])[0]
            recall = precision_recall_curve(y_test_classes == i, y_pred_proba[:, i])[1]
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
        
    except Exception as e:
        logging.error(f"Erro ao avaliar modelo: {str(e)}")
        raise

def plot_results(model, history, X_test, y_test, feature_cols, window_size):
    """Plota visualizações detalhadas dos resultados"""
    try:
        os.makedirs('ml_model_build_neural_network/figures', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Histórico de treinamento
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title('Loss por Época')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Treino')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title('Acurácia por Época')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['AUC'], label='Treino')
        plt.plot(history.history['val_AUC'], label='Validação')
        plt.title('AUC por Época')
        plt.xlabel('Época')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'ml_model_build_neural_network/figures/training_history_{window_size}_{timestamp}.png')
        plt.close()
        
        # Matriz de confusão
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        plt.figure(figsize=(10, 7))
        cm = confusion_matrix(y_test_classes, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Derrota', 'Empate', 'Vitória'],
            yticklabels=['Derrota', 'Empate', 'Vitória']
        )
        plt.title(f'Matriz de Confusão - Rede Neural - Janela de {window_size} Jogos')
        plt.ylabel('Real')
        plt.xlabel('Previsto')
        plt.savefig(f'ml_model_build_neural_network/figures/confusion_matrix_{window_size}_{timestamp}.png')
        plt.close()
        
        # Feature Importance baseada nos pesos da primeira camada
        weights = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': weights
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Features Mais Importantes - Janela de {window_size} Jogos')
        plt.xlabel('Importância')
        plt.tight_layout()
        plt.savefig(f'ml_model_build_neural_network/figures/feature_importance_{window_size}_{timestamp}.png')
        plt.close()
        
        # Logging das features mais importantes
        logging.info("\nTop 10 Features Mais Importantes:")
        for idx, row in feature_importance.head(10).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        logging.error(f"Erro ao plotar resultados: {str(e)}")
        raise

def save_model(model, scaler, le, window_size):
    """Salva o modelo, scaler e label encoder"""
    try:
        os.makedirs('ml_model_build_neural_network/ml_models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_path = f'ml_model_build_neural_network/ml_models/neural_network_{window_size}_{timestamp}.keras'
        scaler_path = f'ml_model_build_neural_network/ml_models/scaler_{window_size}_{timestamp}.pkl'
        le_path = f'ml_model_build_neural_network/ml_models/label_encoder_{window_size}_{timestamp}.pkl'
        
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)
        
        logging.info(f"Modelo, scaler e label encoder salvos em {model_path}")
        
    except Exception as e:
        logging.error(f"Erro ao salvar modelo: {str(e)}")
        raise

def main():
    print('\n\n ---------------- START ---------------- \n')
    
    try:
        # Criar diretórios necessários
        os.makedirs('ml_model_build_neural_network/figures', exist_ok=True)
        os.makedirs('ml_model_build_neural_network/ml_models', exist_ok=True)
        os.makedirs('ml_model_build_neural_network/logs', exist_ok=True)
        os.makedirs('ml_model_build_neural_network/checkpoints', exist_ok=True)
        
        # Processar ambas as janelas de tempo
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            logging.info(f"Dados carregados com sucesso. Shape: {df.shape}")
            
            X_train, X_test, y_train, y_test, feature_cols, scaler, le = prepare_data(df)
            
            # Criar e treinar modelo
            model = create_model(input_shape=X_train.shape[1])
            model, history = train_model(model, X_train, y_train, X_test, y_test, window_size)
            
            # Avaliar modelo
            y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, window_size, le)
            
            # Plotar resultados
            plot_results(model, history, X_test, y_test, feature_cols, window_size)
            
            # Salvar modelo
            save_model(model, scaler, le, window_size)
            
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main() 