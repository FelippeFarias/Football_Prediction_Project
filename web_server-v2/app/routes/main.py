from flask import Blueprint, render_template, current_app
from app.utils.predictions import load_predictions, clean_past_predictions
import pandas as pd
import os

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    try:
        # Verificar se o arquivo de predições existe
        if not os.path.exists(current_app.config['PREDICTIONS_FILE']):
            current_app.logger.error("Arquivo de predições não encontrado")
            return render_template('error.html', 
                                message="Não há predições disponíveis no momento. Por favor, execute o script de predições primeiro."), 404
        
        # Carregar predições
        pl_pred = load_predictions()
        current_app.logger.info(f"Predições carregadas com sucesso. Shape: {pl_pred.shape}")
        
        # Remover jogos passados
        pl_pred = clean_past_predictions(pl_pred)
        current_app.logger.info(f"Jogos passados removidos. Shape: {pl_pred.shape}")
        
        if pl_pred.empty:
            return render_template('error.html', 
                                message="Não há jogos futuros para exibir. Por favor, atualize as predições."), 404
        
        # Ordenar por data
        pl_pred = pl_pred.sort_values('Game Date')
        
        # Converter datas para formato legível
        pl_pred['Game Date'] = pd.to_datetime(pl_pred['Game Date']).dt.strftime('%d/%m/%Y')
        
        # Converter para dicionário e enviar para o template
        predictions = pl_pred.to_dict('records')
        current_app.logger.info(f"Enviando {len(predictions)} predições para o template")
        
        return render_template('index.html', predictions=predictions)
        
    except Exception as e:
        current_app.logger.error(f"Erro ao carregar predições: {str(e)}")
        return render_template('error.html', 
                             message="Erro ao carregar as predições. Por favor, tente novamente mais tarde."), 500 