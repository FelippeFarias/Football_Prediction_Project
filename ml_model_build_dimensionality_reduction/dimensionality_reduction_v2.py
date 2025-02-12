import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import warnings
from typing import Dict, Any, Tuple
import umap

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print('\n\n ---------------- START ---------------- \n')

def load_data(window_size: int = 10) -> pd.DataFrame:
    """Carrega os dados do arquivo mais recente"""
    try:
        df_files = [f for f in os.listdir('prem_clean_fixtures_and_dataframes') if f.endswith(f'_v2.txt')]
        if not df_files:
            raise FileNotFoundError("Nenhum arquivo de dados encontrado")
        
        df_file = max([f for f in df_files if f'{window_size}_v2.txt' in f], 
                     key=lambda x: os.path.getctime(os.path.join('prem_clean_fixtures_and_dataframes', x)))
        
        with open(f'prem_clean_fixtures_and_dataframes/{df_file}', 'rb') as myFile:
            df = pickle.load(myFile)
            
        logging.info(f"Dados carregados com sucesso para janela de {window_size} jogos")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {str(e)}")
        raise

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepara os dados para redução de dimensionalidade"""
    try:
        feature_cols = [col for col in df.columns if col not in ['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']]
        X = df[feature_cols].values
        y = df['Team Result Indicator'].values
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {str(e)}")
        raise

def apply_pca(X: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """Aplica PCA com análise de componentes"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    return {
        'transformed_data': X_pca,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'components': pca.components_
    }

def apply_kernel_pca(X: np.ndarray, n_components: int = 2, kernel: str = 'rbf') -> np.ndarray:
    """Aplica Kernel PCA para capturar relações não-lineares"""
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    return kpca.fit_transform(X)

def apply_sparse_pca(X: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """Aplica Sparse PCA para seleção de features"""
    spca = SparsePCA(n_components=n_components, random_state=42)
    X_spca = spca.fit_transform(X)
    
    return {
        'transformed_data': X_spca,
        'components': spca.components_
    }

def apply_tsne(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Aplica t-SNE para visualização de alta dimensionalidade"""
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(X)

def apply_umap(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Aplica UMAP para preservação de estrutura local e global"""
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)

def plot_results(results: Dict[str, Any], 
                y: np.ndarray, 
                feature_cols: list,
                window_size: int):
    """Plota visualizações dos resultados da redução de dimensionalidade"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('ml_model_build_dimensionality_reduction/figures', exist_ok=True)
    
    # Plot da variância explicada do PCA
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(results['explained_variance_ratio']) + 1),
             results['explained_variance_ratio'], 'bo-')
    plt.title('Variância Explicada por Componente')
    plt.xlabel('Componente Principal')
    plt.ylabel('Razão de Variância Explicada')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(results['cumulative_variance_ratio']) + 1),
             results['cumulative_variance_ratio'], 'ro-')
    plt.title('Variância Explicada Acumulada')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    
    plt.tight_layout()
    plt.savefig(f'ml_model_build_dimensionality_reduction/figures/pca_variance_{window_size}_{timestamp}.png')
    plt.close()
    
    # Plot das contribuições das features
    plt.figure(figsize=(12, 6))
    components_df = pd.DataFrame(
        results['components'],
        columns=feature_cols
    )
    
    sns.heatmap(components_df, cmap='coolwarm', center=0)
    plt.title('Contribuição das Features para os Componentes Principais')
    plt.xlabel('Features')
    plt.ylabel('Componentes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'ml_model_build_dimensionality_reduction/figures/feature_contributions_{window_size}_{timestamp}.png')
    plt.close()
    
    # Plot da projeção 2D
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(results['transformed_data'][:, 0],
                         results['transformed_data'][:, 1],
                         c=y,
                         cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Projeção 2D dos Dados')
    plt.xlabel('Primeira Componente')
    plt.ylabel('Segunda Componente')
    plt.savefig(f'ml_model_build_dimensionality_reduction/figures/projection_2d_{window_size}_{timestamp}.png')
    plt.close()

def save_results(results: Dict[str, Any], window_size: int):
    """Salva os resultados da redução de dimensionalidade"""
    os.makedirs('ml_model_build_dimensionality_reduction/results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_path = f'ml_model_build_dimensionality_reduction/results/dim_reduction_{window_size}_{timestamp}.pkl'
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Resultados salvos em {results_path}")

def main():
    try:
        for window_size in [5, 10]:
            logging.info(f"\nProcessando janela de {window_size} jogos...")
            
            # Carregar e preparar dados
            df = load_data(window_size)
            X, y, feature_cols = prepare_data(df)
            
            # Aplicar diferentes técnicas de redução de dimensionalidade
            results = {}
            
            # PCA
            logging.info("Aplicando PCA...")
            pca_results = apply_pca(X)
            results['pca'] = pca_results
            
            # Kernel PCA
            logging.info("Aplicando Kernel PCA...")
            kpca_results = apply_kernel_pca(X)
            results['kpca'] = kpca_results
            
            # Sparse PCA
            logging.info("Aplicando Sparse PCA...")
            spca_results = apply_sparse_pca(X)
            results['spca'] = spca_results
            
            # t-SNE
            logging.info("Aplicando t-SNE...")
            tsne_results = apply_tsne(X)
            results['tsne'] = tsne_results
            
            # UMAP
            logging.info("Aplicando UMAP...")
            umap_results = apply_umap(X)
            results['umap'] = umap_results
            
            # Plotar resultados
            plot_results(pca_results, y, feature_cols, window_size)
            
            # Salvar resultados
            save_results(results, window_size)
            
            # Logging das conclusões
            n_components_95 = np.argmax(pca_results['cumulative_variance_ratio'] >= 0.95) + 1
            logging.info(f"\nConclusões para janela de {window_size} jogos:")
            logging.info(f"Número de componentes para 95% da variância: {n_components_95}")
            logging.info(f"Variância explicada pelos 2 primeiros componentes: {pca_results['explained_variance_ratio'][:2].sum():.2%}")
            
    except Exception as e:
        logging.error(f"Erro na execução principal: {str(e)}")
        raise
    finally:
        print('\n ----------------- END ----------------- \n')

if __name__ == "__main__":
    main() 