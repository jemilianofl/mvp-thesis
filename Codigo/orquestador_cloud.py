import os
import sys
import subprocess
import logging
from datetime import datetime

# Logging optimizado para la consola de GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class PipelineCloud:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        
        # Inyectamos el día de hoy para que los scripts solo busquen deltas recientes
        self.hoy = datetime.now().strftime("%Y-%m-%d")
        os.environ["FECHA_ACTUALIZACION"] = self.hoy
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        # SECUENCIA CLOUD: Estrictamente lo necesario para la inferencia diaria
        # Omitimos la Fase 0 (La malla ya debe estar en el repo) y la Fase 3, 4 y 5 (Modelos y gráficas)
        self.secuencia = [
            os.path.join("01_Data_Engineering", "01a_scraping_conagua.py"),
            os.path.join("01_Data_Engineering", "01b_correccion_sesgo_daymet.py"),
            os.path.join("02_Feature_Engineering", "02a_extraccion_gee.py"),
            os.path.join("02_Feature_Engineering", "02b_ensamblador_tensor.py"),
            os.path.join("02_Feature_Engineering", "02c_enriquecimiento_tensor.py"),
            os.path.join("02_Feature_Engineering", "02d_mascaras_uso_suelo.py"),
            os.path.join("02_Feature_Engineering", "02e_inyeccion_potencial.py"),
            os.path.join("05_Deep_Learning", "05a_generador_secuencias.py"), # Necesario para armar el numpy 3D
            os.path.join("06_Inference", "06a_ensamble.py"),
            os.path.join("06_Inference", "06a_vigilancia_regional.py")
        ]

    def ejecutar(self):
        logging.info("="*60)
        logging.info(f"☁️ INICIANDO PIPELINE CLOUD MLOPS - FECHA: {self.hoy}")
        logging.info("="*60)
        
        try:
            for ruta_relativa in self.secuencia:
                ruta_absoluta = os.path.join(self.dir_base, ruta_relativa)
                if not os.path.exists(ruta_absoluta):
                    logging.warning(f"⏭️ OMITIDO (No encontrado en repo): {ruta_relativa}")
                    continue
                
                logging.info(f"🟢 EJECUTANDO: {os.path.basename(ruta_absoluta)}")
                # Se utiliza sys.executable para forzar el Python del entorno de GitHub Actions
                subprocess.run([sys.executable, ruta_absoluta], check=True, encoding='utf-8', env=os.environ.copy())

            logging.info("="*60)
            logging.info("✅ PIPELINE CLOUD COMPLETADO. PREDICCIONES ACTUALIZADAS.")
            logging.info("="*60)
            
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ FALLO EN EL PIPELINE CLOUD.")
            sys.exit(1) # Le dice a GitHub Actions que el proceso falló

if __name__ == "__main__":
    PipelineCloud().ejecutar()