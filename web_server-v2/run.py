from flask import Flask, render_template
from app import create_app
import logging
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_server.log'),
        logging.StreamHandler()
    ]
)

# Criar diretório de logs se não existir
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.info("Iniciando servidor web...")
print("Iniciando servidor...")

# Obter diretório base e raiz do projeto
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

print(f"Diretório base: {base_dir}")
print(f"Diretório raiz do projeto: {project_root}")

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 