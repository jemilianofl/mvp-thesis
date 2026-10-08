import os
import sys
import subprocess
import logging
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mlops_pipeline.log", encoding='utf-8')
    ]
)

class PipelineMLOps:
    def __init__(self, modo_reentrenamiento=False):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.modo_reentrenamiento = modo_reentrenamiento
        
        self.hoy = datetime.now().strftime("%Y-%m-%d")
        os.environ["FECHA_ACTUALIZACION"] = self.hoy
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        self.fase_0_malla = [("Malla Regional", "00b_generador_malla.py")]
        
        self.fase_1_data_eng = [
            ("Scraping CONAGUA", os.path.join("01_Data_Engineering", "01a_scraping_conagua.py")),
            ("Corrección Daymet", os.path.join("01_Data_Engineering", "01b_correccion_sesgo_daymet.py")),
            ("Malla Parquet", os.path.join("01_Data_Engineering", "01c_actualizacion_malla_parquet.py")),
            ("Recorte INEGI", os.path.join("01_Data_Engineering", "01d_recorte_inegi.py"))
        ]
        
        # FIX MLOPS: Agregamos "Generador Secuencias" a la rutina diaria obligatoria
        self.fase_2_feature_eng = [
            ("Extracción GEE", os.path.join("02_Feature_Engineering", "02a_extraccion_gee.py")),
            ("Ensamblador Tensor", os.path.join("02_Feature_Engineering", "02b_ensamblador_tensor.py")),
            ("Enriquecimiento", os.path.join("02_Feature_Engineering", "02c_enriquecimiento_tensor.py")),
            ("Máscaras Uso de Suelo", os.path.join("02_Feature_Engineering", "02d_mascaras_uso_suelo.py")),
            ("Inyección Potencial", os.path.join("02_Feature_Engineering", "02e_inyeccion_potencial.py")),
            ("Generador Secuencias LSTM", os.path.join("05_Deep_Learning", "05a_generador_secuencias.py")) 
        ]
        
        # La fase de entrenamiento ahora solo contiene los motores pesados
        self.fase_3_5_entrenamiento = [
            ("Motor Isolation Forest", os.path.join("03_Baseline_Models", "03a_motor_isolation_forest.py")),
            ("Motor LOF", os.path.join("03_Baseline_Models", "03b_motor_lof.py")),
            ("Ensamble Evolutivo", os.path.join("03_Baseline_Models", "03c_ensamble_evolutivo.py")),
            ("Entrenamiento LSTM", os.path.join("05_Deep_Learning", "05b_modelo_lstm.py"))
        ]
        
        self.fase_6_inferencia = [
            ("Ensamble de Inferencia", os.path.join("06_Inference", "06a_ensamble.py")),
            ("Vigilancia Regional LSTM", os.path.join("06_Inference", "06a_vigilancia_regional.py")),
            ("Validación Física (Ground Truth)", os.path.join("06_Inference", "06b_validacion_fisica.py"))
        ]

    def ejecutar_script(self, nombre, ruta_relativa):
        ruta_absoluta = os.path.join(self.dir_base, ruta_relativa)
        if not os.path.exists(ruta_absoluta):
            logging.warning(f"⏭️ SCRIPT OMITIDO (No encontrado): {ruta_absoluta}")
            return

        logging.info(f"🟢 INICIANDO: {nombre}")
        try:
            resultado = subprocess.run(
                [sys.executable, ruta_absoluta], 
                check=True, capture_output=True, text=True, encoding='utf-8', env=os.environ.copy()
            )
            for linea in resultado.stdout.split('\n'):
                if linea.strip(): logging.info(f"   {linea.strip()}")
            logging.info(f"✅ {nombre.upper()} COMPLETADO.\n")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ ERROR CRÍTICO EN {nombre.upper()}:")
            logging.error(e.stderr.strip() if e.stderr else e.output)
            raise Exception(f"Pipeline detenido por fallo en {nombre}")

    def ejecutar_pipeline(self):
        logging.info("="*70)
        logging.info(f"🚀 INICIANDO PIPELINE MLOPS - FECHA DE CORTE: {self.hoy}")
        logging.info("="*70)
        inicio = time.time()
        
        try:
            secuencia = self.fase_0_malla + self.fase_1_data_eng + self.fase_2_feature_eng
            if self.modo_reentrenamiento:
                logging.info("⚠️ MODO REENTRENAMIENTO ACTIVADO")
                secuencia += self.fase_3_5_entrenamiento
            else:
                logging.info("⚡ MODO VIGILANCIA DIARIA (Solo Ingesta + Inferencia)")
                
            secuencia += self.fase_6_inferencia
            for nombre, ruta in secuencia:
                self.ejecutar_script(nombre, ruta)
                
            fin = time.time()
            logging.info("="*70)
            logging.info(f"🏆 PIPELINE MLOPS COMPLETADO EXITOSAMENTE en {round((fin - inicio)/60, 2)} min.")
            logging.info("="*70)
            
        except Exception as e:
            logging.critical(f"🛑 PIPELINE ABORTADO: {str(e)}")

if __name__ == "__main__":
    PipelineMLOps(modo_reentrenamiento=False).ejecutar_pipeline()