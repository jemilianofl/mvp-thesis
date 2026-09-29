import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

class TemporalHeatmapPipeline:
    """
    Pipeline MLOps para Validación Macro.
    Genera un Heatmap Espacio-Temporal para observar el comportamiento 
    sincronizado de todos los polígonos de la región de estudio.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_ensamble"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        os.makedirs(self.reports_folder, exist_ok=True)

    def generar_macro_heatmap(self):
        print("\n🗺️ [ETAPA 1] Procesando Matrices Espacio-Temporales...")
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        
        if not archivos:
            print("❌ No se encontraron predicciones del ensamble.")
            return
            
        lista_dfs = []
        for f in archivos:
            pol_id = os.path.basename(f).replace("ensamble_preds_", "").replace(".parquet", "")
            df = pd.read_parquet(f)
            df['POLIGONO'] = f"Polígono {pol_id}"
            lista_dfs.append(df[['FECHA_MES', 'POLIGONO', 'NDVI', 'ENSAMBLE_PRED']])
            
        df_master = pd.concat(lista_dfs, ignore_index=True)
        df_master['FECHA_MES'] = pd.to_datetime(df_master['FECHA_MES']).dt.strftime('%Y-%m')
        
        # 1. Crear matriz pivot para el NDVI
        matriz_ndvi = df_master.pivot(index='POLIGONO', columns='FECHA_MES', values='NDVI')
        
        # 2. Crear matriz pivot para las Anomalías Confirmadas
        matriz_anomalias = df_master.pivot(index='POLIGONO', columns='FECHA_MES', values='ENSAMBLE_PRED')
        
        print("\n📊 [ETAPA 2] Generando Gráfica de Heatmap Académico...")
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="white")
        
        fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
        
        # Usamos una paleta divergente donde el verde es selva sana y el café/rojo es degradación
        cmap = sns.diverging_palette(30, 130, s=90, l=45, as_cmap=True)
        
        sns.heatmap(matriz_ndvi, cmap=cmap, cbar_kws={'label': 'Salud de la Vegetación (NDVI)'}, 
                    yticklabels=True, xticklabels=12, linewidths=0.5, linecolor='white', ax=ax)
        
        # Superponer las anomalías del ensamble con una máscara (solo dibujar donde hay anomalía == -1)
        import numpy as np
        mascara_anomalias = np.where(matriz_anomalias == -1, 1, np.nan)
        sns.heatmap(mascara_anomalias, mask=np.isnan(mascara_anomalias), cmap=['black'], 
                    cbar=False, annot=False, ax=ax, alpha=0.8)
        
        plt.title('Dinámica Forestal Regional: Historial NDVI y Detección de Colapsos (Ensamble 10D)', 
                  fontsize=18, fontweight='bold', pad=20, color='black')
        plt.xlabel('Línea de Tiempo (Año-Mes)', fontsize=14, color='black', labelpad=10)
        plt.ylabel('Polígonos de Entrenamiento (Selva)', fontsize=14, color='black')
        
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=11)
        
        # Leyenda personalizada para los parches negros
        from matplotlib.patches import Patch
        leyenda_anomalia = Patch(facecolor='black', label='Colapso Confirmado por Ensamble')
        ax.legend(handles=[leyenda_anomalia], loc='lower right', bbox_to_anchor=(1.0, -0.15), 
                  frameon=True, facecolor='white', edgecolor='black', fontsize=12)
        
        plt.tight_layout()
        ruta_grafica16 = os.path.join(self.reports_folder, "16_Heatmap_Temporal_Macro.png")
        plt.savefig(ruta_grafica16, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Heatmap temporal exportado a: {self.reports_folder}")

if __name__ == "__main__":
    pipeline = TemporalHeatmapPipeline()
    pipeline.generar_macro_heatmap()