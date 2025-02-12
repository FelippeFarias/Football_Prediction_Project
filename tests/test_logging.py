import pytest
import logging
import os
from datetime import datetime
from ml_functions.logging_functions import (
    setup_logger,
    log_model_performance,
    log_prediction,
    log_error
)

@pytest.fixture
def test_dir(tmpdir):
    """Fixture para criar diretório temporário para testes"""
    test_dir = tmpdir.mkdir('test_logs')
    return str(test_dir)

@pytest.fixture
def sample_logger(test_dir):
    """Fixture para criar logger de teste"""
    log_file = os.path.join(test_dir, 'test.log')
    return setup_logger(
        name='test_logger',
        log_file=log_file,
        level=logging.INFO
    )

@pytest.mark.logging
def test_setup_logger(test_dir):
    """Testa a configuração do logger"""
    log_file = os.path.join(test_dir, 'test.log')
    logger = setup_logger(
        name='test_logger',
        log_file=log_file,
        level=logging.INFO
    )
    
    # Verificar se o logger foi criado corretamente
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert os.path.exists(log_file)
    
    # Verificar se os handlers foram configurados
    assert len(logger.handlers) == 2  # FileHandler e StreamHandler
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    # Testar logging
    test_message = 'Test message'
    logger.info(test_message)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content

@pytest.mark.logging
def test_log_model_performance(sample_logger, test_dir):
    """Testa o logging de performance do modelo"""
    metrics = {
        'accuracy': 0.75,
        'precision': 0.73,
        'recall': 0.72,
        'f1': 0.74,
        'roc_auc': 0.80
    }
    
    log_model_performance(
        logger=sample_logger,
        model_name='RandomForest',
        metrics=metrics,
        window_size=10
    )
    
    log_file = sample_logger.handlers[0].baseFilename
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # Verificar se as informações foram registradas
        assert 'RandomForest' in log_content
        assert 'window_size: 10' in log_content
        for metric, value in metrics.items():
            assert f'{metric}: {value}' in log_content

@pytest.mark.logging
def test_log_prediction(sample_logger, test_dir):
    """Testa o logging de predições"""
    home_team = 'TeamA'
    away_team = 'TeamB'
    probabilities = {
        'Vitória': 0.5,
        'Empate': 0.3,
        'Derrota': 0.2
    }
    date = '2024-02-08'
    
    log_prediction(
        logger=sample_logger,
        home_team=home_team,
        away_team=away_team,
        probabilities=probabilities,
        date=date
    )
    
    log_file = sample_logger.handlers[0].baseFilename
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # Verificar se as informações foram registradas
        assert home_team in log_content
        assert away_team in log_content
        assert date in log_content
        for outcome, prob in probabilities.items():
            assert f'{outcome}: {prob:.2%}' in log_content

@pytest.mark.logging
def test_log_error(sample_logger, test_dir):
    """Testa o logging de erros"""
    try:
        raise ValueError('Test error')
    except Exception as e:
        log_error(
            logger=sample_logger,
            error=e,
            context='test_context'
        )
    
    log_file = sample_logger.handlers[0].baseFilename
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # Verificar se as informações foram registradas
        assert 'Erro em test_context' in log_content
        assert 'ValueError' in log_content
        assert 'Test error' in log_content
        assert 'Stack trace completo' in log_content

@pytest.mark.logging
def test_logger_formatting(test_dir):
    """Testa a formatação personalizada do logger"""
    log_file = os.path.join(test_dir, 'format_test.log')
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = setup_logger(
        name='format_test',
        log_file=log_file,
        level=logging.INFO,
        format_str=format_str
    )
    
    test_message = 'Formatted test message'
    logger.info(test_message)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # Verificar formato da mensagem
        assert ' - format_test - INFO - ' in log_content
        assert test_message in log_content

@pytest.mark.logging
def test_logger_levels(test_dir):
    """Testa diferentes níveis de logging"""
    log_file = os.path.join(test_dir, 'levels_test.log')
    logger = setup_logger(
        name='levels_test',
        log_file=log_file,
        level=logging.INFO
    )
    
    # Testar diferentes níveis
    debug_msg = 'Debug message'
    info_msg = 'Info message'
    warning_msg = 'Warning message'
    error_msg = 'Error message'
    
    logger.debug(debug_msg)
    logger.info(info_msg)
    logger.warning(warning_msg)
    logger.error(error_msg)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # DEBUG não deve aparecer (nível INFO)
        assert debug_msg not in log_content
        # Outros níveis devem aparecer
        assert info_msg in log_content
        assert warning_msg in log_content
        assert error_msg in log_content 