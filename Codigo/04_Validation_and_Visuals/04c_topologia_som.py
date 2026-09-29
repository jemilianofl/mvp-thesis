import os
import glob
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import warnings

warnings.filterwarnings("ignore")

class MapaInteractivoPipeline:
    """
    Pipeline MLOps para la generación del Entregable Geográfico Final.
    Cruza los resultados del Ensamble de Modelos con las geometrías originales 
    para crear un mapa web interactivo de alertas tempranas.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_ensamble = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "04_Resultados", "baseline_ensamble"))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        os.makedirs(self.reports_folder, exist_ok=True)

    def compilar_resultados_geograficos(self):
        print("\n🗺️ [ETAPA 1] Compilando Resultados Espacio-Temporales...")
        if not os.path.exists(self.ruta_poligonos):
            print("❌ Archivo maestro de polígonos no encontrado.")
            return None
            
        gdf_poligonos = gpd.read_file(self.ruta_poligonos)
        archivos_ensamble = glob.glob(os.path.join(self.dir_ensamble, "*.parquet"))
        
        if not archivos_ensamble:
            print("❌ No se encontraron resultados del modelo de ensamble.")
            return None

        resumen_anomalias = {}
        
        for f in archivos_ensamble:
            pol_id = os.path.basename(f).replace("ensamble_preds_", "").replace(".parquet", "")
            df = pd.read_parquet(f)
            
            total_meses = len(df)
            anomalias_confirmadas = (df['ENSAMBLE_PRED'] == -1).sum()
            ndvi_promedio = df['NDVI'].mean()
            
            resumen_anomalias[pol_id] = {
                'Total_Anomalias': int(anomalias_confirmadas),
                'Meses_Evaluados': int(total_meses),
                'NDVI_Promedio': round(float(ndvi_promedio), 3),
                'Estado_Salud': 'Crítico' if anomalias_confirmadas > 2 else ('Alerta' if anomalias_confirmadas > 0 else 'Estable')
            }
            
        poligonos_validos = list(resumen_anomalias.keys())
        
        if 'ID_POLIGONO' in gdf_poligonos.columns:
            gdf_final = gdf_poligonos[gdf_poligonos['ID_POLIGONO'].astype(str).isin(poligonos_validos)].copy()
            gdf_final['ID_MATCH'] = gdf_final['ID_POLIGONO'].astype(str)
        else:
            gdf_final = gdf_poligonos[gdf_poligonos.index.astype(str).isin(poligonos_validos)].copy()
            gdf_final['ID_MATCH'] = gdf_final.index.astype(str)
            
        gdf_final['Total_Anomalias'] = gdf_final['ID_MATCH'].map(lambda x: resumen_anomalias[x]['Total_Anomalias'])
        gdf_final['NDVI_Promedio'] = gdf_final['ID_MATCH'].map(lambda x: resumen_anomalias[x]['NDVI_Promedio'])
        gdf_final['Estado_Salud'] = gdf_final['ID_MATCH'].map(lambda x: resumen_anomalias[x]['Estado_Salud'])
        
        return gdf_final

    def generar_mapa_html(self, gdf):
        print("\n🌐 [ETAPA 2] Renderizando Mapa Web Interactivo...")
        
        centro_lat = 19.5
        centro_lon = -89.0
        
        # Mapa base forzado a Esri Light Gray (Sin API keys, sin bloqueos)
        mapa = folium.Map(
            location=[centro_lat, centro_lon], 
            zoom_start=7, 
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
        )
        
        def estilo_poligono(feature):
            estado = feature['properties']['Estado_Salud']
            if estado == 'Crítico':
                color = '#d73027' 
            elif estado == 'Alerta':
                color = '#fdae61' 
            else:
                color = '#1a9850' 
                
            return {
                'fillColor': color,
                'color': 'black',
                'weight': 1.5,
                'fillOpacity': 0.75
            }

        folium.GeoJson(
            gdf,
            style_function=estilo_poligono,
            tooltip=folium.features.GeoJsonTooltip(
                fields=['ID_MATCH', 'Estado_Salud', 'Total_Anomalias', 'NDVI_Promedio'],
                aliases=['ID Polígono:', 'Diagnóstico:', 'Colapsos Históricos:', 'NDVI Promedio:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 13px; padding: 10px; border: 1px solid black;")
            )
        ).add_to(mapa)
        
        plugins.Fullscreen(position='topright').add_to(mapa)
        
        ruta_html = os.path.join(self.reports_folder, "17_Mapa_Interactivo_Anomalias.html")
        mapa.save(ruta_html)
        
        print(f"   ✅ Mapa interactivo guardado exitosamente en: {self.reports_folder}")
        print("   👉 Abre el archivo '17_Mapa_Interactivo_Anomalias.html' directamente en tu navegador web.")

    def ejecutar(self):
        gdf_resultados = self.compilar_resultados_geograficos()
        if gdf_resultados is not None and not gdf_resultados.empty:
            self.generar_mapa_html(gdf_resultados)

if __name__ == "__main__":
    pipeline = MapaInteractivoPipeline()
    pipeline.ejecutar()