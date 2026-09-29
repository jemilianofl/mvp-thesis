import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class IsolationForestBaselinePipeline:
    """
    Pipeline MLOps para el Modelo Base No Supervisado (Isolation Forest).
    Detecta anomalías multivariables (posible deforestación o degradación) 
    analizando el comportamiento histórico de la selva.
    """
    
    def __init__(self):
        # Rutas de directorios
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_iforest"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        # Hiperparámetros del Modelo
        self.contamination = 0.05  # Asumimos que el 5% de las observaciones históricas son anomalías graves
        self.random_state = 42

    def entrenar_y_predecir(self):
        print("\n🌲 [ETAPA 1] Entrenando Motor de Isolation Forest...")
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        if not archivos: 
            print("❌ No se encontraron tensores filtrados en 03_Model_Ready.")
            return False
            
        # Definir las variables que el modelo va a analizar para buscar anomalías
        features = [
            'NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 
            'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 
            'ESTRES_HIDRICO', 'ELEVACION',
            'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE'
        ]
        
        exitos = 0
        for archivo in tqdm(archivos, desc="Procesando Polígonos"):
            nombre_archivo = os.path.basename(archivo).replace("tensor_modelo_", "iforest_preds_")
            df = pd.read_parquet(archivo)
            
            # Verificar que las columnas existan
            cols_disponibles = [col for col in features if col in df.columns]
            if not cols_disponibles or df.empty: continue
                
            # Extraer matriz de entrenamiento
            X = df[cols_disponibles].fillna(0).values
            
            # Inicializar y Entrenar Isolation Forest
            modelo_if = IsolationForest(
                n_estimators=100, 
                max_samples='auto', 
                contamination=self.contamination, 
                random_state=self.random_state
            )
            
            # Predicción: 1 (Normal), -1 (Anomalía)
            df['IFOREST_PRED'] = modelo_if.fit_predict(X)
            
            # Score de anomalía (más negativo = más anómalo)
            df['IFOREST_SCORE'] = modelo_if.decision_function(X)
            
            # Guardar resultados
            df.to_parquet(os.path.join(self.dir_salida, nombre_archivo), index=False)
            exitos += 1
            
        print(f"   ✅ Detección completada en {exitos} polígonos.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica de Detección de Anomalías...")
        archivos = glob.glob(os.path.join(self.dir_salida, "*.parquet"))
        if not archivos: return
        
        # Tomar el primer archivo que tenga anomalías detectadas para el reporte
        df = None
        for f in archivos:
            temp_df = pd.read_parquet(f)
            if (temp_df['IFOREST_PRED'] == -1).sum() > 0:
                df = temp_df
                break
                
        if df is None:
            df = pd.read_parquet(archivos[0]) # Fallback si no hay anomalías
            
        df['FECHA_MES'] = pd.to_datetime(df['FECHA_MES'])
        
        # Estándar visual académico (Limpieza de caché de Matplotlib)
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        # Línea normal del NDVI
        ax.plot(df['FECHA_MES'], df['NDVI'], color='#2ca02c', linewidth=2, label='Salud de la Selva (NDVI)', zorder=1)
        
        # Filtrar las anomalías
        anomalias = df[df['IFOREST_PRED'] == -1]
        
        # Marcar las anomalías con puntos rojos
        ax.scatter(anomalias['FECHA_MES'], anomalias['NDVI'], color='#d62728', s=100, marker='X', edgecolor='black', 
                   label='Anomalía Detectada (Posible Deforestación/Incendio)', zorder=5)
        
        ax.set_title('Detección No Supervisada de Anomalías (Isolation Forest)', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_xlabel('Tiempo', fontsize=12, color='black')
        ax.set_ylabel('Índice de Vegetación (NDVI)', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=10)
        
        ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='black', fontsize=11)
        
        ruta_grafica11 = os.path.join(self.reports_folder, "11_Deteccion_Anomalias_IsolationForest.png")
        plt.savefig(ruta_grafica11, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()
        print(f"   ✅ Gráfica del modelo exportada a: {self.reports_folder}")

    def ejecutar(self):
        if self.entrenar_y_predecir():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = IsolationForestBaselinePipeline()
    pipeline.ejecutar()