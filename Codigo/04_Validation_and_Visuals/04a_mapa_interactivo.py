import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class SOMAnalysisPipeline:
    """
    Pipeline MLOps para Redes Neuronales No Supervisadas (Self-Organizing Maps).
    Proyecta el hiperespacio de 12 variables continuas en mapas topológicos 2D.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_finales"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        os.makedirs(self.reports_folder, exist_ok=True)
        
        # Las 12 dimensiones exactas (Ahora incluyendo Wavelets)
        self.features = [
            'NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 
            'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 
            'ESTRES_HIDRICO', 'ELEVACION',
            'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE'
        ]
        
        self.som_grid_size = 15 

    def cargar_y_escalar_datos(self):
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        if not archivos: return None, None
        
        df_list = [pd.read_parquet(f) for f in archivos]
        df_master = pd.concat(df_list, ignore_index=True)
        
        df_master = df_master.dropna(subset=self.features).reset_index(drop=True)
        X = df_master[self.features].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, df_master

    def ejecutar(self):
        print("\n🧠 [ETAPA 1] Entrenando Red Neuronal SOM (Self-Organizing Map)...")
        X_scaled, _ = self.cargar_y_escalar_datos()
        
        if X_scaled is None:
            print("❌ No se encontraron tensores en 03_Model_Ready/tensores_finales.")
            return

        som = MiniSom(x=self.som_grid_size, y=self.som_grid_size, input_len=len(self.features), 
                      sigma=1.5, learning_rate=0.5, random_seed=42)
        som.pca_weights_init(X_scaled)
        
        print("   ► Ajustando topología (10,000 iteraciones)...")
        som.train_batch(X_scaled, 10000, verbose=True)

        print("\n📊 [ETAPA 2] Generando Feature Planes (Visualización 12D -> 2D)...")
        W = som.get_weights() 

        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="white")
        
        # SOLUCIÓN: Cuadrícula de 3 filas por 4 columnas (12 gráficas)
        fig, axes = plt.subplots(3, 4, figsize=(20, 14), facecolor='white')
        axes = axes.flatten()

        for i, feature in enumerate(self.features):
            ax = axes[i]
            peso_variable = W[:, :, i].T
            
            c = ax.pcolor(peso_variable, cmap='viridis', edgecolors='w', linewidths=0.1)
            ax.set_title(feature, fontsize=12, fontweight='bold', color='black')
            ax.set_xticks([])
            ax.set_yticks([])
            
            cbar = fig.colorbar(c, ax=ax, orientation='horizontal', pad=0.03, aspect=40)
            cbar.ax.tick_params(labelsize=8)

        plt.suptitle('Topología SOM: Mapas de Componentes (Hiperespacio 12D a Plano 2D)', 
                     fontsize=18, fontweight='bold', color='black', y=1.02)
        
        plt.tight_layout()
        ruta_grafica15 = os.path.join(self.reports_folder, "15_Topologia_SOM_Feature_Planes.png")
        plt.savefig(ruta_grafica15, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Mapas neuronales topológicos exportados a: {self.reports_folder}")

if __name__ == "__main__":
    pipeline = SOMAnalysisPipeline()
    pipeline.ejecutar()