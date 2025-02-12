import logging
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logger(name: str, log_file: str, level: int = logging.INFO, format_str: str = None) -> logging.Logger:
    """
    Configura um logger com handlers para arquivo e console

    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log
        level: Nível de logging
        format_str: String de formatação personalizada

    Returns:
        Logger configurado
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Criar diretório de logs se não existir
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configurar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Criar formatador
    formatter = logging.Formatter(format_str)

    # Handler para arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def log_model_performance(logger: logging.Logger,
                         model_name: str,
                         metrics: Dict[str, float],
                         window_size: int) -> None:
    """
    Registra métricas de performance do modelo
    
    Args:
        logger: Logger configurado
        model_name: Nome do modelo
        metrics: Dicionário com métricas
        window_size: Tamanho da janela
    """
    logger.info('\n' + '=' * 50)
    logger.info(f'Performance do Modelo {model_name}')
    logger.info(f'window_size: {window_size}')
    logger.info(f'Data/Hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 50)
    
    for metric, value in metrics.items():
        logger.info(f'{metric}: {value:.2f}')
        
    logger.info('=' * 50 + '\n')

def log_prediction(logger: logging.Logger,
                  home_team: str,
                  away_team: str,
                  probabilities: Dict[str, float],
                  date: str) -> None:
    """
    Registra predições no log

    Args:
        logger: Logger configurado
        home_team: Time da casa
        away_team: Time visitante
        probabilities: Probabilidades preditas
        date: Data do jogo
    """
    logger.info("\n==================================================")
    logger.info(f"Predição para {home_team} vs {away_team}")
    logger.info(f"Data: {date}")
    logger.info("==================================================")
    
    for result, prob in probabilities.items():
        logger.info(f"{result}: {prob:.2%}")
    
    logger.info("==================================================\n")

def log_error(logger: logging.Logger,
              error: Exception,
              context: str) -> None:
    """
    Registra erro ocorrido
    
    Args:
        logger: Logger configurado
        error: Exceção ocorrida
        context: Contexto do erro
    """
    logger.error('\n' + '!' * 50)
    logger.error(f'Erro em {context}')
    logger.error(f'Data/Hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.error('!' * 50)
    
    logger.error(f'Tipo: {type(error).__name__}')
    logger.error(f'Mensagem: {str(error)}')
    logger.error('Traceback:')
    logger.error(traceback.format_exc())
    logger.error('!' * 50 + '\n') 