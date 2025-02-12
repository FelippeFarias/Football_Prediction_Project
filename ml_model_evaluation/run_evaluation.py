import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model_build_random_forest.random_forest_model_build import load_data, prepare_data
from model_evaluator import ModelEvaluator
import logging
import joblib

def get_latest_model(window_size):
    """Encontra o modelo mais recente para um determinado tamanho de janela"""
    model_dir = 'ml_model_build_random_forest/ml_models'
    model_files = [f for f in os.listdir(model_dir) 
                  if f.startswith(f'random_forest_model_{window_size}_') and f.endswith('.pkl')]
    
    if not model_files:
        return None
        
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

def main():
    try:
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_model_evaluation/evaluation.log'),
                logging.StreamHandler()
            ]
        )
        
        # Criar diretórios necessários
        os.makedirs('ml_model_evaluation/figures', exist_ok=True)
        os.makedirs('ml_model_evaluation/results', exist_ok=True)
        
        # Carregar dados
        window_sizes = [5, 10]  # Apenas janelas disponíveis
        evaluator = ModelEvaluator(test_size=0.2, validation_size=0.2)
        
        for window_size in window_sizes:
            logging.info(f"\nAvaliando modelo com janela de {window_size} jogos")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            X, y, feature_cols, scaler, le = prepare_data(df)
            
            # Dividir dados em treino, validação e teste
            X_train, X_val, X_test, y_train, y_val, y_test = evaluator.split_data(X, y)
            
            # Carregar modelo treinado mais recente
            model_path = get_latest_model(window_size)
            if model_path is None:
                logging.error(f"Nenhum modelo encontrado para janela de {window_size} jogos")
                continue
                
            logging.info(f"Usando modelo: {os.path.basename(model_path)}")
            try:
                model = joblib.load(model_path)
                logging.info("Modelo carregado com sucesso")
            except Exception as e:
                logging.error(f"Erro ao carregar modelo: {str(e)}")
                continue
            
            # Avaliar no conjunto de validação
            logging.info("\nAvaliação no conjunto de validação:")
            y_val_pred = model.predict(X_val)
            y_val_pred_proba = model.predict_proba(X_val)
            val_results = evaluator.evaluate_predictions(y_val, y_val_pred, y_val_pred_proba)
            
            # Avaliar no conjunto de teste
            logging.info("\nAvaliação no conjunto de teste:")
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)
            test_results = evaluator.evaluate_predictions(y_test, y_test_pred, y_test_pred_proba)
            
            # Plotar e salvar resultados
            evaluator.plot_evaluation_results(val_results, window_size)
            evaluator.plot_evaluation_results(test_results, window_size)
            
            evaluator.save_results({
                'validation': val_results,
                'test': test_results,
                'window_size': window_size,
                'feature_cols': feature_cols,
                'model_used': os.path.basename(model_path)
            }, window_size)
            
            logging.info(f"\nAvaliação completa para janela de {window_size} jogos")
            
    except Exception as e:
        logging.error(f"Erro durante a avaliação: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 