import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class LOFBaselinePipeline:
    """
    Pipeline MLOps para el Modelo Base No Supervisado (Local Outlier Factor - LOF).
    Detecta anomalías evaluando la desviación de la densidad local de un mes 
    respecto a sus vecinos temporales.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_lof"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        self.contamination = 0.05
        self.n_neighbors = 20 # Ventana de meses vecinos a evaluar

    def entrenar_y_predecir(self):
        print("\n🔍 [ETAPA 1] Entrenando Motor de Local Outlier Factor (LOF)...")
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        if not archivos: 
            print("❌ No se encontraron tensores en 03_Model_Ready.")
            return False
            
        features = [
            'NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 
            'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 
            'ESTRES_HIDRICO', 'ELEVACION',
            'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE'
        ]
        exitos = 0
        
        for archivo in tqdm(archivos, desc="Procesando Polígonos con LOF"):
            nombre_archivo = os.path.basename(archivo).replace("tensor_modelo_", "lof_preds_")
            df = pd.read_parquet(archivo)
            
            cols_disponibles = [col for col in features if col in df.columns]
            if not cols_disponibles or df.empty: continue
                
            X = df[cols_disponibles].fillna(0).values
            
            # LOF para detección de novedades/anomalías
            modelo_lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X) - 1), 
                contamination=self.contamination,
                novelty=False
            )
            
            df['LOF_PRED'] = modelo_lof.fit_predict(X)
            df['LOF_SCORE'] = modelo_lof.negative_outlier_factor_
            
            df.to_parquet(os.path.join(self.dir_salida, nombre_archivo), index=False)
            exitos += 1
            
        print(f"   ✅ Detección LOF completada en {exitos} polígonos.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica de Anomalías por Densidad Local (LOF)...")
        archivos = glob.glob(os.path.join(self.dir_salida, "*.parquet"))
        if not archivos: return
        
        df = None
        for f in archivos:
            temp_df = pd.read_parquet(f)
            if (temp_df['LOF_PRED'] == -1).sum() > 0:
                df = temp_df
                break
                
        if df is None: df = pd.read_parquet(archivos[0])
            
        df['FECHA_MES'] = pd.to_datetime(df['FECHA_MES'])
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        ax.plot(df['FECHA_MES'], df['NDVI'], color='#1f77b4', linewidth=2, label='Salud de la Selva (NDVI)', zorder=1)
        
        anomalias = df[df['LOF_PRED'] == -1]
        ax.scatter(anomalias['FECHA_MES'], anomalias['NDVI'], color='#ff7f0e', s=100, marker='D', edgecolor='black', 
                   label='Anomalía LOF (Cambio Brusco Local)', zorder=5)
        
        ax.set_title('Detección de Anomalías por Densidad Local (LOF)', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_xlabel('Tiempo', fontsize=12, color='black')
        ax.set_ylabel('Índice de Vegetación (NDVI)', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=10)
        
        ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='black', fontsize=11)
        
        ruta_grafica12 = os.path.join(self.reports_folder, "12_Deteccion_Anomalias_LOF.png")
        plt.savefig(ruta_grafica12, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()
        print(f"   ✅ Gráfica LOF exportada a: {self.reports_folder}")

    def ejecutar(self):
        if self.entrenar_y_predecir():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = LOFBaselinePipeline()
    pipeline.ejecutar()