import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class EnsambleEvolutivoPipeline:
    """
    Pipeline MLOps para Ensamblar Modelos Base (Isolation Forest + LOF).
    Utiliza votación estricta (Hard Voting) para confirmar anomalías solo 
    cuando ambos algoritmos independientes coinciden, reduciendo falsos positivos.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_iforest = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_iforest"))
        self.dir_lof = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_lof"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_ensamble"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)

    def ejecutar_ensamble(self):
        print("\n🤝 [ETAPA 1] Ejecutando Ensamble de Modelos Base (Hard Voting)...")
        archivos_if = glob.glob(os.path.join(self.dir_iforest, "*.parquet"))
        
        if not archivos_if:
            print("❌ No se encontraron resultados de Isolation Forest.")
            return False
            
        exitos = 0
        
        for arch_if in tqdm(archivos_if, desc="Ensamblando Predicciones"):
            nombre_base = os.path.basename(arch_if).replace("iforest_preds_", "")
            arch_lof = os.path.join(self.dir_lof, f"lof_preds_{nombre_base}")
            
            if not os.path.exists(arch_lof): continue
                
            df_if = pd.read_parquet(arch_if)
            df_lof = pd.read_parquet(arch_lof)
            
            # Unir las predicciones de ambos modelos
            df_ensamble = df_if.copy()
            df_ensamble['LOF_PRED'] = df_lof['LOF_PRED']
            
            # Lógica de Votación Estricta: Anomalía (-1) solo si AMBOS dicen -1. Si no, Normal (1).
            df_ensamble['ENSAMBLE_PRED'] = df_ensamble.apply(
                lambda row: -1 if (row['IFOREST_PRED'] == -1 and row['LOF_PRED'] == -1) else 1, axis=1
            )
            
            nombre_salida = f"ensamble_preds_{nombre_base}"
            df_ensamble.to_parquet(os.path.join(self.dir_salida, nombre_salida), index=False)
            exitos += 1
            
        print(f"   ✅ Ensamble completado para {exitos} polígonos.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica de Consenso de Anomalías...")
        archivos = glob.glob(os.path.join(self.dir_salida, "*.parquet"))
        if not archivos: return
        
        # Buscar el mejor ejemplo para la tesis (un polígono con anomalías confirmadas)
        df_plot = None
        for f in archivos:
            df = pd.read_parquet(f)
            if (df['ENSAMBLE_PRED'] == -1).sum() > 0:
                df_plot = df
                break
                
        if df_plot is None: df_plot = pd.read_parquet(archivos[0])
            
        df_plot['FECHA_MES'] = pd.to_datetime(df_plot['FECHA_MES'])
        
        # Estándar Académico Blanco
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        # Línea de salud forestal
        ax.plot(df_plot['FECHA_MES'], df_plot['NDVI'], color='#2ca02c', linewidth=2.5, label='Salud de la Selva (NDVI)', zorder=1)
        
        # Filtrar los puntos donde AMBOS modelos coinciden
        anomalias_consensuadas = df_plot[df_plot['ENSAMBLE_PRED'] == -1]
        
        ax.scatter(anomalias_consensuadas['FECHA_MES'], anomalias_consensuadas['NDVI'], 
                   color='#9467bd', s=130, marker='*', edgecolor='black', 
                   label='Anomalía Confirmada (Consenso IF + LOF)', zorder=5)
        
        ax.set_title('Consenso Analítico de Degradación Forestal (Modelo Ensamble)', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_xlabel('Mes de Observación', fontsize=12, color='black')
        ax.set_ylabel('Índice de Salud de Vegetación (NDVI)', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=10)
        
        ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='black', fontsize=11)
        
        ruta_grafica14 = os.path.join(self.reports_folder, "14_Anomalias_Consenso_Ensamble.png")
        plt.savefig(ruta_grafica14, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ Gráfica de consenso exportada a: {self.reports_folder}")

    def ejecutar(self):
        if self.ejecutar_ensamble():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = EnsambleEvolutivoPipeline()
    pipeline.ejecutar()