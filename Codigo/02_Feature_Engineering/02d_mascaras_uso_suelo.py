import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore")

class LandUseMaskingPipeline:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_enriquecido"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        self.auditoria = {'Polígonos Totales': 0, 'Selva (Monitoreo)': 0, 'Otros Usos (Grises)': 0}

    def aplicar_mascaras(self):
        print("\n🛡️ [ETAPA 1] Aplicando Máscaras No Destructivas (Etiquetado Inteligente)...")
            
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        self.auditoria['Polígonos Totales'] = len(archivos)
        
        for archivo in tqdm(archivos, desc="Etiquetando Coberturas"):
            df = pd.read_parquet(archivo)
            if df.empty: continue
            
            ndvi_maximo_historico = df['NDVI'].max() if 'NDVI' in df.columns else 0
            
            # FIX NO DESTRUCTIVO: En lugar de NO exportar los menores a 0.45, les ponemos una bandera.
            # ES_SELVA = 1 significa que la IA debe evaluarlo. ES_SELVA = 0 significa que la IA lo ignora, PERO conserva sus variables.
            if ndvi_maximo_historico >= 0.45:
                df['ES_SELVA'] = 1
                self.auditoria['Selva (Monitoreo)'] += 1
            else:
                df['ES_SELVA'] = 0
                self.auditoria['Otros Usos (Grises)'] += 1
                
            nombre_salida = os.path.basename(archivo).replace("tensor_enriquecido", "tensor_modelo")
            df.to_parquet(os.path.join(self.dir_salida, nombre_salida), index=False)
                
        print(f"   ✅ Depuración finalizada. {self.auditoria['Selva (Monitoreo)']} polígonos listos para IA, {self.auditoria['Otros Usos (Grises)']} mantenidos en monitoreo pasivo.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica Académica de Depuración del Dataset...")
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        
        categorias = ['Retenidos (Selva)', 'Descartados (Ruido/Urbano)']
        valores = [self.auditoria['Selva (Monitoreo)'], self.auditoria['Otros Usos (Grises)']]
        colores = ['#41ab5d', '#ef3b2c'] 
        
        bars = ax.bar(categorias, valores, color=colores, width=0.6, edgecolor='black', linewidth=1.2)
        
        ax.set_title('Mapeo del Dataset por Cobertura Forestal\n(Máscara de Uso de Suelo)', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_ylabel('Cantidad de Polígonos', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=11)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
                        
        texto_total = f"Polígonos Orginales Preservados: {self.auditoria['Polígonos Totales']}"
        plt.figtext(0.15, 0.80, texto_total, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'), fontsize=11, color='black')

        ruta_grafica10 = os.path.join(self.reports_folder, "10_Depuracion_Uso_Suelo.png")
        plt.savefig(ruta_grafica10, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ Gráfica de depuración exportada a: {self.reports_folder}")

    def ejecutar(self):
        if self.aplicar_mascaras():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = LandUseMaskingPipeline()
    pipeline.ejecutar()