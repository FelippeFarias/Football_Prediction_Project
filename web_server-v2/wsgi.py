import sys
import os

# Adiciona o diretório atual ao PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("Iniciando servidor...")
    print(f"PYTHONPATH: {sys.path}")
    print(f"Diretório atual: {os.getcwd()}")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Erro ao iniciar o servidor: {str(e)}")
        import traceback
        traceback.print_exc() 