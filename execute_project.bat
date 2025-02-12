@echo off

REM Definindo o diretório raiz do projeto
set PROJECT_ROOT=%CD%

REM Criando diretórios necessários
mkdir "%PROJECT_ROOT%\prem_clean_fixtures_and_dataframes" 2>nul
mkdir "%PROJECT_ROOT%\predictions" 2>nul
mkdir "%PROJECT_ROOT%\predictions\logs" 2>nul
mkdir "%PROJECT_ROOT%\predictions\results" 2>nul
mkdir "%PROJECT_ROOT%\predictions\figures" 2>nul
mkdir "%PROJECT_ROOT%\ml_model_build_random_forest\models" 2>nul
mkdir "%PROJECT_ROOT%\ml_model_build_ensemble_model\models" 2>nul
mkdir "%PROJECT_ROOT%\web_server-v2\logs" 2>nul

REM Ativando o ambiente virtual
call .venv\Scripts\activate

REM Adicionando o diretório raiz ao PYTHONPATH
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

REM Executando scripts de coleta e preparação de dados
python "%PROJECT_ROOT%\01_api_data_request.py"
python "%PROJECT_ROOT%\02_cleaning_stats_data.py"
python "%PROJECT_ROOT%\03_feature_engineering.py"

REM Treinando modelos de machine learning
cd "%PROJECT_ROOT%\ml_model_build_random_forest"
python "%PROJECT_ROOT%\ml_model_build_random_forest\random_forest_model_build.py"
cd "%PROJECT_ROOT%"

cd "%PROJECT_ROOT%\ml_model_build_ensemble_model"
python "%PROJECT_ROOT%\ml_model_build_ensemble_model\ensemble_model_build.py"
cd "%PROJECT_ROOT%"

REM Gerando previsões
cd "%PROJECT_ROOT%\predictions"
python "%PROJECT_ROOT%\predictions\predictions_v2.py"
cd "%PROJECT_ROOT%"

REM Iniciando o servidor web
cd "%PROJECT_ROOT%\web_server-v2"
python "%PROJECT_ROOT%\web_server-v2\run.py"

pause