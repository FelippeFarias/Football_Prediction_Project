from flask import render_template, jsonify, request
from app import app
from app.config.config import Config
import pandas as pd
import pickle
import logging
import os
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_predictions():
    """Carrega as predições mais recentes"""
    try:
        predictions_dir = os.path.join(Config.PROJECT_ROOT, 'predictions', 'results')
        prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.pkl')]
        
        if not prediction_files:
            logging.warning("Nenhum arquivo de predições encontrado")
            return pd.DataFrame()
            
        # Pegar o arquivo mais recente
        latest_file = max(prediction_files, key=lambda x: os.path.getctime(os.path.join(predictions_dir, x)))
        file_path = os.path.join(predictions_dir, latest_file)
        
        with open(file_path, 'rb') as f:
            predictions = pickle.load(f)
            
        # Converter datas para datetime
        predictions['Game Date'] = pd.to_datetime(predictions['Game Date'])
        return predictions
        
    except Exception as e:
        logging.error(f"Erro ao carregar predições: {str(e)}")
        return pd.DataFrame()

def load_additional_stats():
    """Carrega estatísticas adicionais dos times"""
    try:
        with open(Config.ADDITIONAL_STATS_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Erro ao carregar estatísticas adicionais: {str(e)}")
        return {}

@app.route('/')
@app.route('/index')
def index():
    try:
        predictions_df = load_predictions()
        if predictions_df.empty:
            return render_template('error.html', message="Não foi possível carregar as predições")
            
        # Ordenar por data
        predictions_df = predictions_df.sort_values('Game Date')
        
        # Formatar dados para o template
        games = []
        for _, row in predictions_df.iterrows():
            game = {
                'home_team': row['Home Team'],
                'away_team': row['Away Team'],
                'home_win_prob': f"{row['Home Win']:.1f}%",
                'draw_prob': f"{row['Draw']:.1f}%",
                'away_win_prob': f"{row['Away Win']:.1f}%",
                'date': row['Game Date'].strftime('%d/%m/%Y'),
                'venue': row.get('Venue', 'N/A'),
                'home_logo': row.get('Home Team Logo', ''),
                'away_logo': row.get('Away Team Logo', '')
            }
            games.append(game)
            
        return render_template('index.html', games=games[:Config.MAX_DISPLAY_GAMES])
        
    except Exception as e:
        logging.error(f"Erro na rota index: {str(e)}")
        return render_template('error.html', message="Erro ao carregar a página")

@app.route('/api/predictions')
def get_predictions():
    try:
        predictions_df = load_predictions()
        if predictions_df.empty:
            return jsonify({'error': 'Não foi possível carregar as predições'}), 404
            
        # Converter para formato JSON
        predictions_list = []
        for _, row in predictions_df.iterrows():
            prediction = {
                'home_team': row['Home Team'],
                'away_team': row['Away Team'],
                'home_win_probability': float(row['Home Win']),
                'draw_probability': float(row['Draw']),
                'away_win_probability': float(row['Away Win']),
                'game_date': row['Game Date'].strftime('%Y-%m-%d'),
                'venue': row.get('Venue', 'N/A'),
                'home_team_logo': row.get('Home Team Logo', ''),
                'away_team_logo': row.get('Away Team Logo', '')
            }
            predictions_list.append(prediction)
            
        return jsonify({'predictions': predictions_list})
        
    except Exception as e:
        logging.error(f"Erro na API de predições: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/team_stats/<team_name>')
def get_team_stats(team_name):
    try:
        stats_dict = load_additional_stats()
        if not stats_dict or team_name not in stats_dict:
            return jsonify({'error': 'Time não encontrado'}), 404
            
        team_stats = stats_dict[team_name]
        
        # Processar estatísticas do time
        processed_stats = []
        for fixture_id, stats in team_stats.items():
            if isinstance(stats, pd.DataFrame):
                stats_dict = stats.to_dict('records')[0]
                processed_stats.append({
                    'fixture_id': fixture_id,
                    'stats': stats_dict
                })
                
        return jsonify({
            'team_name': team_name,
            'stats': processed_stats
        })
        
    except Exception as e:
        logging.error(f"Erro ao buscar estatísticas do time: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message="Página não encontrada"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Erro interno do servidor"), 500 