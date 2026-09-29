import os
import glob
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class TensorEnrichmentPipeline:
    """
    Pipeline MLOps para Ingeniería de Variables Avanzada.
    Inyecta rezagos, ventanas móviles y Transformadas Wavelet (DWT) 
    para extraer anomalías de alta frecuencia en múltiples dimensiones.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_maestro"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_enriquecido"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        os.makedirs(self.dir_salida, exist_ok=True)

    def _extraer_detalle_wavelet(self, senal, wavelet='db2'):
        """
        Aplica Transformada Wavelet Discreta para aislar anomalías de alta frecuencia.
        Retorna la señal reconstruida únicamente con los coeficientes de detalle.
        """
        if len(senal) < 4 or np.isnan(senal).any():
            return np.zeros(len(senal))
            
        cA, cD = pywt.dwt(senal, wavelet)
        detalle_reconstruido = pywt.idwt(None, cD, wavelet)
        return detalle_reconstruido[:len(senal)]
        
    def enriquecer_tensores(self):
        print("\n⚙️ [ETAPA 1] Enriqueciendo Tensores Multidimensionales (Wavelets & Lags)...")
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        if not archivos: return False
        
        exitos = 0
        for archivo in tqdm(archivos, desc="Calculando Espacio de Frecuencias"):
            nombre_archivo = os.path.basename(archivo).replace("tensor_maestro_", "tensor_enriquecido_")
            df = pd.read_parquet(archivo)
            if df.empty: continue
            
            df['FECHA_MES'] = pd.to_datetime(df['FECHA_MES'])
            df = df.sort_values('FECHA_MES').reset_index(drop=True)
            
            # SOLUCIÓN: Interpolar solo las columnas numéricas
            cols_num = df.select_dtypes(include=[np.number]).columns
            df[cols_num] = df[cols_num].interpolate(method='linear').bfill().ffill()
            
            # 1. Transformada Wavelet en Múltiples Dimensiones
            df['NDVI_WAVELET_DETALLE'] = self._extraer_detalle_wavelet(df['NDVI'].values)
            df['PRECIP_WAVELET_DETALLE'] = self._extraer_detalle_wavelet(df['PRECIP_SUM_MES'].values)
            
            if 'BSI' in df.columns:
                df['BSI_WAVELET_DETALLE'] = self._extraer_detalle_wavelet(df['BSI'].values)
            
            # 2. Rezagos (Lags Temporales Clásicos)
            df['NDVI_LAG_1'] = df['NDVI'].shift(1)
            df['NDVI_LAG_3'] = df['NDVI'].shift(3)
            
            # 3. Ventanas Móviles Climáticas
            df['PRECIP_ACUM_3M'] = df['PRECIP_SUM_MES'].rolling(window=3, min_periods=1).sum()
            df['ESTRES_HIDRICO'] = df['TMAX_MEDIAN'] / (df['PRECIP_SUM_MES'] + 1)
            
            df = df.dropna().reset_index(drop=True)
            df.to_parquet(os.path.join(self.dir_salida, nombre_archivo), index=False)
            exitos += 1
            
        print(f"   ✅ {exitos} Tensores enriquecidos con energía Wavelet listos.")
        return True
        
    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráficas de Análisis (Clima y Wavelets)...")
        archivos = glob.glob(os.path.join(self.dir_salida, "*.parquet"))
        if not archivos: return
        
        df = pd.read_parquet(archivos[0])
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # =====================================================================
        # GRÁFICA 09a: EVOLUCIÓN CLIMÁTICA (NDVI vs PRECIPITACIÓN)
        # =====================================================================
        fig1, ax1 = plt.subplots(figsize=(12, 6), facecolor='white')
        
        color_ndvi = '#2ca02c' 
        ax1.set_xlabel('Mes de Observación', fontsize=12, color='black')
        ax1.set_ylabel('Índice de Salud Forestal (NDVI)', color=color_ndvi, fontsize=12, fontweight='bold')
        ax1.plot(df['FECHA_MES'], df['NDVI'], color=color_ndvi, linewidth=2.5, label='NDVI Mensual')
        ax1.tick_params(axis='y', labelcolor=color_ndvi)
        ax1.tick_params(axis='x', colors='black')
        
        ax2 = ax1.twinx()  
        color_precip = '#1f77b4' 
        ax2.set_ylabel('Lluvia Acumulada a 3 Meses (mm)', color=color_precip, fontsize=12, fontweight='bold')
        ax2.fill_between(df['FECHA_MES'], 0, df['PRECIP_ACUM_3M'], color=color_precip, alpha=0.25)
        ax2.tick_params(axis='y', labelcolor=color_precip)
        
        plt.title('Dinámica Eco-Climática Temporal\nRelación entre Precipitación Acumulada y Salud de la Selva', fontsize=14, fontweight='bold', pad=15, color='black')
        
        from matplotlib.patches import Patch
        import matplotlib.lines as mlines
        linea_ndvi = mlines.Line2D([], [], color=color_ndvi, linewidth=2.5, label='Salud Forestal (NDVI)')
        parche_precip = Patch(facecolor=color_precip, alpha=0.25, label='Lluvia Acumulada (3M)')
        ax1.legend(handles=[linea_ndvi, parche_precip], loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=11)
        
        fig1.tight_layout()
        ruta_grafica9a = os.path.join(self.reports_folder, "09a_Evolucion_Temporal_Clima.png")
        plt.savefig(ruta_grafica9a, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)

        # =====================================================================
        # GRÁFICA 09b: DESCOMPOSICIÓN WAVELET (NDVI vs ANOMALÍA ALTA FRECUENCIA)
        # =====================================================================
        fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor='white')
        
        ax3.set_xlabel('Mes de Observación', fontsize=12, color='black')
        ax3.set_ylabel('Señal Original (NDVI)', color=color_ndvi, fontsize=12, fontweight='bold')
        ax3.plot(df['FECHA_MES'], df['NDVI'], color=color_ndvi, linewidth=2, label='NDVI Bruto')
        ax3.tick_params(axis='y', labelcolor=color_ndvi)
        
        ax4 = ax3.twinx()  
        color_wave = '#d62728' 
        ax4.set_ylabel('Coeficiente de Detalle Wavelet (Anomalía)', color=color_wave, fontsize=12, fontweight='bold')
        ax4.plot(df['FECHA_MES'], df['NDVI_WAVELET_DETALLE'], color=color_wave, linewidth=1.5, linestyle='--', label='Alta Frecuencia (Wavelet)')
        ax4.tick_params(axis='y', labelcolor=color_wave)
        
        ax4.axhline(0, color='black', linewidth=1, alpha=0.5)
        
        plt.title('Descomposición de Señal (Transformada Wavelet Discreta)\nAislamiento de Cambios Abruptos en la Serie de Tiempo', fontsize=14, fontweight='bold', pad=15, color='black')
        
        fig3.tight_layout()
        ruta_grafica9b = os.path.join(self.reports_folder, "09b_Wavelet_Transform_Analysis.png")
        plt.savefig(ruta_grafica9b, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        
        print(f"   ✅ Gráficas climáticas y de espectro Wavelet exportadas a: {self.reports_folder}")

    def ejecutar(self):
        if self.enriquecer_tensores():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = TensorEnrichmentPipeline()
    pipeline.ejecutar()