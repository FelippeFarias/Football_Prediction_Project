from flask import Flask, render_template
from app.config.config import Config
import logging
import os

# Criar diretório de logs se não existir
if not os.path.exists('logs'):
    os.makedirs('logs')

# Criar diretório static se não existir
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_server.log'),
        logging.StreamHandler()
    ]
)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Registrando blueprints
    from app.routes import main
    app.register_blueprint(main.bp)
    
    # Importar rotas após a criação da aplicação
    from app import routes

    # Configurar manipuladores de erro
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', message="Página não encontrada"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('error.html', message="Erro interno do servidor"), 500
    
    return app 