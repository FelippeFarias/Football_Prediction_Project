# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:21:32 2020

@author: mhayt
"""

print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import time
start=time.time()

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps  # Nova importação para usar colormaps
import logging
import os
from datetime import datetime
import seaborn as sns

plt.close('all')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#------------------------------- INPUT VARIABLES ------------------------------

# Find the latest ML dataframes
try:
    df_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') if f.endswith('_v2.txt')]
    if not df_files:
        raise FileNotFoundError("No ML dataframe files found")
    
    df_5_files = [f for f in df_files if '_5_v2.txt' in f]
    df_10_files = [f for f in df_files if '_10_v2.txt' in f]
    
    if not df_5_files or not df_10_files:
        raise FileNotFoundError("Missing either 5-game or 10-game window files")
    
    df_5_saved_name = max(df_5_files, key=lambda x: os.path.getctime(os.path.join('prem_clean_fixtures_and_dataframes', x)))
    df_10_saved_name = max(df_10_files, key=lambda x: os.path.getctime(os.path.join('prem_clean_fixtures_and_dataframes', x)))
    
    logging.info(f"Using 5-game window file: {df_5_saved_name}")
    logging.info(f"Using 10-game window file: {df_10_saved_name}")
except Exception as e:
    logging.error(f"Error finding ML dataframe files: {str(e)}")
    raise

# Visualization settings
save_figures = True
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

colourbar = 'winter'
transparency = 0.6
markersize = 25

#-------------------------- PRE-ML DATA VISUALISATION -------------------------

try:
    # Load dataframes
    with open(f'prem_clean_fixtures_and_dataframes/{df_5_saved_name}', 'rb') as myFile:
        df_ml_5 = pickle.load(myFile)
        logging.info("Loaded 5-game window dataframe")
        logging.info(f"Columns in 5-game window dataframe: {df_ml_5.columns.tolist()}")

    with open(f'prem_clean_fixtures_and_dataframes/{df_10_saved_name}', 'rb') as myFile:
        df_ml_10 = pickle.load(myFile)
        logging.info("Loaded 10-game window dataframe")
        logging.info(f"Columns in 10-game window dataframe: {df_ml_10.columns.tolist()}")

    #---------- DATA PREP ----------

    # Verificar valores únicos antes da filtragem
    result_column = 'Result Indicator'
    unique_results_5 = df_ml_5[result_column].unique()
    unique_results_10 = df_ml_10[result_column].unique()
    
    logging.info(f"Unique values in 5-game window {result_column}: {unique_results_5}")
    logging.info(f"Unique values in 10-game window {result_column}: {unique_results_10}")
    
    # Ajustar plot_results baseado nos valores reais
    plot_results = sorted(list(set(unique_results_5) | set(unique_results_10)))
    logging.info(f"Adjusted plot_results to match available data: {plot_results}")
    
    # Filter results based on plot_results
    df_ml_10 = df_ml_10[df_ml_10[result_column].isin(plot_results)].reset_index(drop=True)
    df_ml_5 = df_ml_5[df_ml_5[result_column].isin(plot_results)].reset_index(drop=True)
    
    logging.info(f"Filtered to show results: {plot_results}")
    logging.info(f"Number of matches in 5-game window: {len(df_ml_5)}")
    logging.info(f"Number of matches in 10-game window: {len(df_ml_10)}")

    if len(df_ml_5) == 0 and len(df_ml_10) == 0:
        raise ValueError("No data available after filtering. Please check the result indicators in your data.")

    #----------------------------------- FIGURES ---------------------------------
    
    def create_feature_plots(df: pd.DataFrame, window_size: int) -> plt.Figure:
        """
        Cria gráficos de análise para features relevantes
        
        Args:
            df: DataFrame com os dados
            window_size: Tamanho da janela
            
        Returns:
            Figura com os gráficos
        """
        # Criar figura com subplots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # 1. Distribuição de Resultados
        ax1 = fig.add_subplot(gs[0, 0])
        result_counts = df['Result Indicator'].value_counts().sort_index()
        sns.barplot(x=['Derrota', 'Empate', 'Vitória'], y=result_counts.values, ax=ax1)
        ax1.set_title('Distribuição dos Resultados')
        ax1.set_ylabel('Número de Jogos')
        
        # 2. Gols vs Resultado
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(x='Result Indicator', y='Goals Mean', data=df, ax=ax2)
        ax2.set_title('Média de Gols por Resultado')
        ax2.set_xticklabels(['Derrota', 'Empate', 'Vitória'])
        
        # 3. Correlação entre Features Principais
        ax3 = fig.add_subplot(gs[1, :])
        features = ['Goals Mean', 'Shots on Goal Mean', 'Ball Possession Mean', 'Total passes Mean']
        correlation_matrix = df[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax3)
        ax3.set_title('Correlação entre Features Principais')
        
        # 4. Shots on Goal vs Goals com Resultado
        ax4 = fig.add_subplot(gs[2, 0])
        scatter = ax4.scatter(df['Shots on Goal Mean'], 
                             df['Goals Mean'],
                             c=df['Result Indicator'],
                             cmap='coolwarm',
                             alpha=0.6)
        ax4.set_xlabel('Média de Chutes ao Gol')
        ax4.set_ylabel('Média de Gols')
        ax4.set_title('Relação entre Chutes ao Gol e Gols')
        legend = ax4.legend(*scatter.legend_elements(), title="Resultado")
        ax4.add_artist(legend)
        
        # 5. Posse de Bola vs Passes com Resultado
        ax5 = fig.add_subplot(gs[2, 1])
        scatter = ax5.scatter(df['Ball Possession Mean'],
                             df['Total passes Mean'],
                             c=df['Result Indicator'],
                             cmap='coolwarm',
                             alpha=0.6)
        ax5.set_xlabel('Média de Posse de Bola (%)')
        ax5.set_ylabel('Média de Passes Totais')
        ax5.set_title('Relação entre Posse de Bola e Passes')
        legend = ax5.legend(*scatter.legend_elements(), title="Resultado")
        ax5.add_artist(legend)
        
        # Ajustar layout
        plt.suptitle(f'Análise de Features para Predição de Resultados (Janela de {window_size} Jogos)', 
                    fontsize=16, y=0.95)
        plt.tight_layout()
        
        return fig
    
    # Create plots
    if len(df_ml_10) > 0:
        fig_10 = create_feature_plots(df_ml_10, 10)
        if save_figures:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_10.savefig(f'{figures_dir}/average_10_games_{timestamp}.png')
            logging.info(f"Saved 10-game window plot")
            
    if len(df_ml_5) > 0:
        fig_5 = create_feature_plots(df_ml_5, 5)
        if save_figures:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_5.savefig(f'{figures_dir}/average_5_games_{timestamp}.png')
            logging.info(f"Saved 5-game window plot")
    
    plt.show()

except Exception as e:
    logging.error(f"Error during visualization: {str(e)}")
    raise

finally:
    print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
    print(' ----------------- END ----------------- \n')
