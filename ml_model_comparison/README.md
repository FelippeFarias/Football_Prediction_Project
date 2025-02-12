# Comparação de Modelos de Machine Learning para Previsão de Resultados de Futebol

Este módulo realiza uma comparação abrangente entre diferentes modelos de machine learning para previsão de resultados de jogos de futebol.

## Modelos Avaliados

1. Random Forest
2. Neural Network (MLP)
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. XGBoost
6. LightGBM

## Estrutura do Projeto

```
ml_model_comparison/
├── model_comparison.py     # Script principal
├── requirements.txt       # Dependências
├── results/              # Resultados das comparações
├── figures/             # Visualizações geradas
└── models/              # Modelos treinados
```

## Como Usar

1. Instale as dependências:
```bash
pip install -r ml_model_comparison/requirements.txt
```

2. Execute o script de comparação:
```bash
python ml_model_comparison/model_comparison.py
```

## Resultados

O script irá gerar:

1. Métricas de performance para cada modelo:
   - Acurácia
   - Precisão
   - Recall
   - F1-Score
   - Matriz de Confusão

2. Visualizações:
   - Gráfico comparativo de métricas
   - Matrizes de confusão para cada modelo

3. Análise de retorno esperado para apostas

## Arquivos de Saída

- `results/summary_[timestamp].txt`: Resumo detalhado dos resultados
- `results/comparison_results_[timestamp].pkl`: Resultados completos em formato pickle
- `figures/model_comparison_[timestamp].png`: Visualização comparativa
- `figures/confusion_matrix_[model]_[timestamp].png`: Matrizes de confusão

## Notas

- Os modelos são avaliados usando janelas de 5 e 10 jogos
- Os dados são balanceados usando SMOTEENN
- Todos os modelos são otimizados para performance 