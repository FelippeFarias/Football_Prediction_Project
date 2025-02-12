# Modelo Deep Learning para Previsão de Resultados de Futebol

Este módulo implementa uma rede neural profunda para previsão de resultados de partidas de futebol da Premier League.

## Estrutura do Modelo

O modelo utiliza uma arquitetura de rede neural com as seguintes características:

- Camadas totalmente conectadas (Dense layers)
- Normalização em lote (Batch Normalization)
- Ativação ReLU
- Dropout para regularização
- Arquitetura adaptativa ao número de features de entrada
- 3 classes de saída (Derrota, Empate, Vitória)

## Características Principais

1. **Pré-processamento de Dados**
   - Normalização com StandardScaler
   - Balanceamento de classes com SMOTEENN
   - Suporte para diferentes tamanhos de janela (5 e 10 jogos)

2. **Treinamento**
   - Early Stopping para prevenir overfitting
   - Learning Rate Scheduling
   - Validação cruzada com split 60/20/20 (treino/validação/teste)
   - Batch size de 32
   - Otimizador Adam

3. **Avaliação**
   - Métricas completas por classe
   - Matriz de confusão
   - Cálculo de retorno esperado para apostas
   - Visualizações do processo de treinamento

## Estrutura de Diretórios

```
ml_model_build_deep_learning/
├── deep_learning_model_build_v2.py  # Script principal
├── requirements.txt                 # Dependências
├── README.md                       # Esta documentação
├── ml_models/                      # Modelos treinados
├── figures/                        # Visualizações geradas
└── logs/                          # Logs de treinamento
```

## Como Usar

1. **Instalação**
   ```bash
   pip install -r ml_model_build_deep_learning/requirements.txt
   ```

2. **Execução**
   ```bash
   python ml_model_build_deep_learning/deep_learning_model_build_v2.py
   ```

3. **Resultados**
   - Os modelos treinados são salvos em `ml_models/`
   - As visualizações são salvas em `figures/`
   - Os logs são salvos em `logs/`

## Arquivos Gerados

Para cada execução, são gerados os seguintes arquivos (com timestamp):

1. **Modelos**
   - `deep_learning_model_{window_size}_{timestamp}.pt`
   - `scaler_{window_size}_{timestamp}.pkl`
   - `label_encoder_{window_size}_{timestamp}.pkl`
   - `model_info_{window_size}_{timestamp}.json`

2. **Visualizações**
   - `training_history_{window_size}_{timestamp}.png`
   - `confusion_matrix_{window_size}_{timestamp}.png`

3. **Logs**
   - `training.log`

## Métricas Avaliadas

- Acurácia global
- Precision, Recall e F1-Score por classe
- Matriz de confusão
- Retorno esperado para apostas (quando precision > 0.6)

## Notas de Implementação

1. **Tratamento de Dados**
   - Valores ausentes são preenchidos com 0
   - Features são normalizadas
   - Classes são balanceadas com SMOTEENN

2. **Otimizações**
   - Early stopping com paciência de 7 épocas
   - Redução do learning rate em platôs
   - Dropout de 0.3 para regularização

3. **Requisitos de Hardware**
   - Suporte a GPU via CUDA (se disponível)
   - Fallback para CPU quando GPU não disponível

## Resultados Esperados

O modelo é avaliado em duas configurações:
- Janela de 5 jogos
- Janela de 10 jogos

Para cada configuração, são gerados:
1. Métricas de performance
2. Visualizações de treinamento
3. Análise de retorno esperado para apostas

## Manutenção

O código inclui:
- Logging detalhado
- Tratamento de exceções
- Documentação das funções
- Tipagem estática para principais funções 