import os
import glob
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc  

warnings.filterwarnings("ignore")

class TensorEnrichmentPipeline:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_maestro"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_enriquecido"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        # AUTO-LIMPIEZA: Elimina archivos de corridas anteriores para evitar mezclas
        try:
            [os.remove(f) for f in glob.glob(os.path.join(self.dir_salida, "*.parquet"))]
        except Exception:
            pass

    def _extraer_detalle_wavelet(self, senal, wavelet='db2'):
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
            try:
                nombre_archivo = os.path.basename(archivo).replace("tensor_maestro_", "tensor_enriquecido_")
                df = pd.read_parquet(archivo)
                if df.empty: continue
                
                col_fecha = 'FECHA_MES' if 'FECHA_MES' in df.columns else 'FECHA'
                col_precip = 'PRECIP_SUM_MES' if 'PRECIP_SUM_MES' in df.columns else 'PRECIP'
                col_tmax = 'TMAX_MEDIAN' if 'TMAX_MEDIAN' in df.columns else 'TMAX'
                
                df[col_fecha] = pd.to_datetime(df[col_fecha])
                df = df.sort_values(col_fecha).reset_index(drop=True)
                
                cols_num = df.select_dtypes(include=[np.number]).columns
                if len(cols_num) > 0:
                    df[cols_num] = df[cols_num].interpolate(method='linear').bfill().ffill()
                
                if 'NDVI' in df.columns:
                    df['NDVI_WAVELET_DETALLE'] = self._extraer_detalle_wavelet(df['NDVI'].values)
                    df['NDVI_LAG_1'] = df['NDVI'].shift(1)
                    df['NDVI_LAG_3'] = df['NDVI'].shift(3)
                
                if col_precip in df.columns:
                    df['PRECIP_WAVELET_DETALLE'] = self._extraer_detalle_wavelet(df[col_precip].values)
                    df['PRECIP_ACUM_3M'] = df[col_precip].rolling(window=3, min_periods=1).sum()
                
                if 'BSI' in df.columns:
                    df['BSI_WAVELET_DETALLE'] = self._extraer_detalle_wavelet(df['BSI'].values)
                
                if col_tmax in df.columns and col_precip in df.columns:
                    df['ESTRES_HIDRICO'] = df[col_tmax] / (df[col_precip] + 1)
                
                # FIX CRÍTICO: Rellenar con ceros en lugar de borrar la fila (dropna) para salvar la península
                df = df.fillna(0).reset_index(drop=True)
                df.to_parquet(os.path.join(self.dir_salida, nombre_archivo), index=False)
                exitos += 1
                
            except Exception:
                pass
            finally:
                try: del df
                except NameError: pass
                gc.collect()
            
        print(f"   ✅ {exitos} Tensores enriquecidos con energía Wavelet listos.")
        return True
        
    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generación de gráficas Wavelet (Proceso optimizado)...")
        pass

    def ejecutar(self):
        if self.enriquecer_tensores():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = TensorEnrichmentPipeline()
    pipeline.ejecutar()