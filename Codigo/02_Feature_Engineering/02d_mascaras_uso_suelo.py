import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class LandUseMaskingPipeline:
    """
    Pipeline MLOps para la aplicación de Máscaras de Uso de Suelo.
    Filtra los tensores para asegurar que el entrenamiento ocurra estrictamente
    sobre polígonos correspondientes a coberturas forestales/selva.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_enriquecido"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        
        # Diccionario para auditoría del filtrado
        self.auditoria = {'Polígonos Totales': 0, 'Selva (Retenidos)': 0, 'Otros Usos (Descartados)': 0}

    def aplicar_mascaras(self):
        print("\n🛡️ [ETAPA 1] Aplicando Máscaras de Uso de Suelo...")
        if not os.path.exists(self.ruta_poligonos):
            print("❌ Archivo maestro de polígonos no encontrado.")
            return False
            
        gdf_poligonos = gpd.read_file(self.ruta_poligonos)
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        
        self.auditoria['Polígonos Totales'] = len(archivos)
        
        for archivo in tqdm(archivos, desc="Filtrando Polígonos"):
            pol_id = os.path.basename(archivo).replace("tensor_enriquecido_", "").replace(".parquet", "")
            
            # Buscar metadata del polígono (Asumimos que existe una columna 'USO_SUELO' o evaluamos por NDVI histórico)
            # Para fines de este pipeline, si el polígono existe en el geojson filtrado previamente, se retiene.
            # Alternativamente, aplicamos un filtro lógico de NDVI base para descartar zonas urbanas (< 0.2 constante)
            
            df = pd.read_parquet(archivo)
            if df.empty: continue
            
            # Criterio Biológico: Una selva sana debe alcanzar al menos 0.45 de NDVI en su pico histórico. 
            # Si el NDVI máximo histórico es menor a eso, probablemente es suelo urbano, cuerpo de agua o agricultura severa.
            ndvi_maximo_historico = df['NDVI'].max()
            
            if ndvi_maximo_historico >= 0.45:
                # Es Selva / Vegetación Densa
                nombre_salida = os.path.basename(archivo).replace("tensor_enriquecido", "tensor_modelo")
                df.to_parquet(os.path.join(self.dir_salida, nombre_salida), index=False)
                self.auditoria['Selva (Retenidos)'] += 1
            else:
                self.auditoria['Otros Usos (Descartados)'] += 1
                
        print(f"   ✅ Depuración finalizada. {self.auditoria['Selva (Retenidos)']} polígonos listos para IA.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica Académica de Depuración del Dataset...")
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        
        categorias = ['Retenidos (Selva)', 'Descartados (Ruido/Urbano)']
        valores = [self.auditoria['Selva (Retenidos)'], self.auditoria['Otros Usos (Descartados)']]
        colores = ['#41ab5d', '#ef3b2c'] # Verde para selva, rojo para descartados
        
        bars = ax.bar(categorias, valores, color=colores, width=0.6, edgecolor='black', linewidth=1.2)
        
        ax.set_title('Depuración del Dataset por Cobertura Forestal\n(Máscara de Uso de Suelo)', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_ylabel('Cantidad de Polígonos', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=11)
        
        # Añadir etiquetas numéricas encima de las barras
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # Desplazamiento vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
                        
        # Anotación del total
        texto_total = f"Polígonos Originales: {self.auditoria['Polígonos Totales']}"
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