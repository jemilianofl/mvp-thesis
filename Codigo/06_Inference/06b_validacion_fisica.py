import os
import glob
import pandas as pd
import geopandas as gpd
import folium
import warnings

warnings.filterwarnings("ignore")

class ValidadorFisico:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.ruta_ensamble = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_ensamble"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "06_Inferencia"))
        
    def generar_mapa_validacion(self, fecha_auditoria):
        print(f"🔬 [ETAPA 1] Cruzando Respuesta Física Real para la fecha: {fecha_auditoria}")
        
        gdf_poligonos = gpd.read_file(self.ruta_poligonos)
        
        archivos = glob.glob(os.path.join(self.ruta_ensamble, "*.parquet"))
        if not archivos:
            print("❌ No se encontró historial real en 04_Resultados.")
            return
            
        df_historial = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
        
        # Filtramos estrictamente por el año y mes que queremos auditar
        df_historial['FECHA_MES'] = df_historial['FECHA_MES'].astype(str)
        df_mes_especifico = df_historial[df_historial['FECHA_MES'].str.startswith(fecha_auditoria)]
        
        gdf_poligonos['ID_POLIGONO'] = gdf_poligonos.index.astype(str)
        df_mes_especifico['ID_POLIGONO'] = df_mes_especifico['ID_POLIGONO'].astype(str)
        
        # Cruzamos las geometrías con los datos de ese mes específico
        gdf_final = gdf_poligonos.merge(df_mes_especifico, on='ID_POLIGONO', how='left')
        
        print(f"🌐 [ETAPA 2] Renderizando Mapa Web de Auditoría ({fecha_auditoria})...")
        m = folium.Map(
            location=[19.5, -89.0], 
            zoom_start=7, 
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
        )
        
        def style_function(feature):
            estado = feature['properties'].get('ENSAMBLE_PRED')
            if estado == -1:
                color = '#d73027' # Rojo (Degradación activa en este mes)
            elif estado == 1:
                color = '#1a9850' # Verde (Sano en este mes)
            else:
                color = '#555555' # Gris (Nublado / Sin datos este mes)
                
            return {
                'fillColor': color,
                'color': '#ffffff',
                'weight': 1.5,
                'fillOpacity': 0.85
            }
            
        columnas_disp = gdf_final.columns.tolist()
        campos_tooltip = ['ID_POLIGONO', 'FECHA_MES', 'ENSAMBLE_PRED']
        alias_tooltip = ['Parcela Control:', 'Mes Auditado:', 'Diagnóstico (-1=Degradación, 1=Sano):']
        
        variables_fisicas = {
            'NDVI': 'NDVI (Vigor):',
            'ESTRES_HIDRICO': 'Estrés Hídrico:',
            'RADAR_VV': 'Radar VV (Estructura Dosel):',
            'RADAR_VH': 'Radar VH (Volumen Ramas):',
            'GCI': 'GCI (Clorofila):',
            'BSI': 'BSI (Suelo Desnudo):',
            'PRECIP_SUM_MES': 'Precipitación Mes (mm):'
        }
        
        for var, alias in variables_fisicas.items():
            if var in columnas_disp:
                gdf_final[var] = pd.to_numeric(gdf_final[var], errors='coerce').fillna(0).round(3)
                campos_tooltip.append(var)
                alias_tooltip.append(alias)
                
        gdf_final['FECHA_MES'] = gdf_final['FECHA_MES'].fillna(f"Sin cobertura en {fecha_auditoria}")
        gdf_final['ENSAMBLE_PRED'] = gdf_final['ENSAMBLE_PRED'].fillna("N/A")
                
        folium.GeoJson(
            gdf_final,
            style_function=style_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=campos_tooltip,
                aliases=alias_tooltip,
                style=("background-color: white; color: black; font-family: arial; font-size: 13px; padding: 10px; border-radius: 4px;")
            )
        ).add_to(m)
        
        nombre_archivo = f"03_Mapa_Validacion_Fisica_{fecha_auditoria.replace('-', '_')}.html"
        ruta_html = os.path.join(self.dir_salida, nombre_archivo)
        m.save(ruta_html)
        print(f"✅ Mapa de auditoría biológica exportado a: {ruta_html}")

if __name__ == "__main__":
    validador = ValidadorFisico()
    
    # -------------------------------------------------------------
    # PRUEBA TEMPORAL: Cambia esta fecha para ver cómo reacciona 
    # la parcela a lo largo de los años en tu tesis.
    # Ejemplos: "2020-01", "2021-06", "2024-05"
    # -------------------------------------------------------------
    MES_A_EVALUAR = "2020-06" 
    
    validador.generar_mapa_validacion(fecha_auditoria=MES_A_EVALUAR)