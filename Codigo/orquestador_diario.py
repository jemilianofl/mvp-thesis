import os
import sys
import time
from datetime import datetime

def ejecutar_script(ruta):
    print(f"\n🟢 INICIANDO: {os.path.basename(ruta)}")
    resultado = os.system(f"{sys.executable} {ruta}")
    if resultado != 0:
        print(f"❌ ERROR CRÍTICO EN {os.path.basename(ruta)}")
        sys.exit(1)
    print(f"✅ COMPLETADO: {os.path.basename(ruta)}")

def pipeline_diario_inferencia():
    print("======================================================================")
    print(f"🚀 INICIANDO PIPELINE DIARIO (SOLO INFERENCIA) - {datetime.now().strftime('%Y-%m-%d')}")
    print("======================================================================")
    start_time = time.time()

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Codigo")
    
    # 1. Extracción e Ingeniería de Datos (Actualiza al día de hoy)
    scripts_datos = [
        "01_Data_Engineering/01a_scraping_conagua.py",
        "01_Data_Engineering/01b_correccion_sesgo_daymet.py",
        "02_Feature_Engineering/02a_extraccion_gee.py",
        "02_Feature_Engineering/02b_ensamblador_tensor.py",
        "02_Feature_Engineering/02c_enriquecimiento_tensor.py"
        # SALTAMOS MÁSCARAS (02d) PARA EVALUAR TODO EL MAPA
    ]
    
    for script in scripts_datos:
        ejecutar_script(os.path.join(base_dir, script))

    # 2. Inferencia Directa (Sin reentrenar)
    scripts_inferencia = [
        "06_Inference/06a_ensamble.py",
        "06_Inference/06a_vigilancia_regional.py"
    ]
    
    for script in scripts_inferencia:
        ejecutar_script(os.path.join(base_dir, script))

    minutos = (time.time() - start_time) / 60
    print("======================================================================")
    print(f"🏆 PIPELINE DIARIO COMPLETADO EXITOSAMENTE en {minutos:.2f} min.")
    print("======================================================================")

if __name__ == "__main__":
    pipeline_diario_inferencia()