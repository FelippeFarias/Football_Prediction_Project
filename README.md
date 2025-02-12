# ML Functions

Biblioteca de funções de Machine Learning para predição de resultados de futebol.

## Instalação

```bash
pip install -e .
```

## Estrutura do Pacote

O pacote está organizado nos seguintes módulos:

### Feature Engineering
- `running_mean`: Calcula média móvel de uma série
- `average_stats_df`: Calcula estatísticas médias dos times
- `mod_df`: Modifica DataFrame para incluir features derivadas

### Preprocessing
- `prepare_features`: Prepara features para treinamento
- `handle_missing_values`: Trata valores faltantes
- `remove_outliers`: Remove outliers usando z-score
- `encode_categorical_features`: Codifica features categóricas
- `create_time_features`: Cria features baseadas em data/hora
- `create_interaction_features`: Cria features de interação

### Modeling
- `optimize_model`: Otimiza hiperparâmetros do modelo
- `train_model`: Treina modelo e avalia performance
- `predict_proba_calibrated`: Faz predições com probabilidades calibradas
- `ensemble_predictions`: Combina predições de múltiplos modelos
- `get_model_explanation`: Gera explicação para as predições

### Evaluation
- `evaluate_classification_metrics`: Calcula métricas de classificação
- `evaluate_betting_metrics`: Calcula métricas para apostas
- `evaluate_calibration`: Avalia calibração das probabilidades
- `evaluate_feature_importance`: Avalia importância das features
- `evaluate_predictions_over_time`: Avalia performance ao longo do tempo

### Prediction
- `make_predictions`: Faz predições usando modelo treinado
- `predict_next_games`: Faz predições para próximos jogos
- `predict_with_ensemble`: Faz predições usando ensemble
- `predict_with_calibration`: Faz predições com calibração
- `get_prediction_explanation`: Gera explicação para predição

### Visualization
- `plot_confusion_matrix`: Plota matriz de confusão
- `plot_roc_curves`: Plota curvas ROC
- `plot_feature_importance`: Plota importância das features
- `plot_prediction_distribution`: Plota distribuição das predições
- `plot_model_comparison`: Plota comparação entre modelos

### Metrics
- `calculate_betting_metrics`: Calcula métricas para apostas
- `calculate_model_metrics`: Calcula métricas do modelo
- `calculate_calibration_metrics`: Calcula métricas de calibração

### Logging
- `setup_logger`: Configura logger personalizado
- `log_model_performance`: Registra performance do modelo
- `log_prediction`: Registra predição individual
- `log_error`: Registra erro com contexto

### Utils
- `load_latest_data`: Carrega dados mais recentes
- `save_model`: Salva modelo e transformadores
- `load_model`: Carrega modelo e transformadores
- `prepare_prediction_data`: Prepara dados para predição
- `format_predictions`: Formata predições em DataFrame
- `calculate_stake`: Calcula stake usando Kelly Criterion
- `get_betting_advice`: Gera recomendações de apostas

## Exemplo de Uso

```python
from ml_functions import (
    prepare_features,
    optimize_model,
    evaluate_betting_metrics,
    plot_confusion_matrix
)

# Preparar dados
X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_features(
    df=data,
    feature_cols=features,
    target_col='Result'
)

# Otimizar modelo
model, best_params = optimize_model(
    model=base_model,
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train
)

# Fazer predições
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Avaliar resultados
metrics = evaluate_betting_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
    odds={'Derrota': 2.5, 'Empate': 3.5, 'Vitória': 2.5}
)

# Visualizar resultados
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    classes=['Derrota', 'Empate', 'Vitória'],
    save_path='confusion_matrix.png'
)
```

## Requisitos

- Python >= 3.8
- NumPy >= 1.26.1
- Pandas >= 2.1.1
- Scikit-learn >= 1.6.1
- Matplotlib >= 3.7.2
- Seaborn >= 0.13.2

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

