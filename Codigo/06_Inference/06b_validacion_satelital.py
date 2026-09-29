import os
import pandas as pd
import geopandas as gpd
import folium
import warnings

warnings.filterwarnings("ignore")

# --- RUTAS MLOPS ---
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_POLIGONOS = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
RUTA_ALERTAS = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "06_Inferencia", "01_Alerta_Deforestacion_Hibrida.csv"))
DIR_SALIDA = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "06_Inferencia"))

def generar_auditoria_satelital():
    print("🛰️ 1/3 Preparando Motor de Validación Satelital...")
    if not os.path.exists(RUTA_ALERTAS):
        print("No hay alertas críticas que revisar.")
        return
        
    df_alertas = pd.read_csv(RUTA_ALERTAS)
    gdf_base = gpd.read_file(RUTA_POLIGONOS)
    gdf_base['ID_POLIGONO'] = gdf_base.index.astype(str)
    
    # Tomar los 5 casos más extremos para auditoría visual
    top_criticos = df_alertas.sort_values(by='PROBABILIDAD_LSTM', ascending=False).head(5)
    
    print(f"📸 2/3 Generando {len(top_criticos)} reportes satelitales para el comité de tesis...")
    
    for idx, row in top_criticos.iterrows():
        id_pol = str(row['ID_POLIGONO'])
        fecha = row['FECHA_MES']
        prob = row['PROBABILIDAD_LSTM'] * 100
        
        # Extraer la geometría exacta de este polígono deforestado
        poligono_geom = gdf_base[gdf_base['ID_POLIGONO'] == id_pol]
        
        if poligono_geom.empty:
            continue
            
        centro_lat = poligono_geom.geometry.centroid.y.values[0]
        centro_lon = poligono_geom.geometry.centroid.x.values[0]
        
        # Crear un mapa focalizado con Zoom extremo (Nivel Dron/Parcela)
        m = folium.Map(location=[centro_lat, centro_lon], zoom_start=16)
        
        # Inyectar imagen de Satélite HD (Esri World Imagery)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satélite Alta Resolución',
            overlay=False
        ).add_to(m)
        
        # Dibujar el borde del polígono en Rojo Alerta
        folium.GeoJson(
            poligono_geom,
            style_function=lambda x: {'color': '#ff0000', 'fillColor': '#ff0000', 'fillOpacity': 0.3, 'weight': 3}
        ).add_to(m)
        
        # Poner un marcador con los datos de la IA
        folium.Marker(
            [centro_lat, centro_lon],
            popup=f"<b>ALERTA CRÍTICA</b><br>Mes: {fecha}<br>Seguridad IA: {prob:.1f}%<br>Colapso de NDVI confirmado."
        ).add_to(m)
        
        nombre_archivo = f"02_Evidencia_Satelital_Pol_{id_pol}_{fecha}.html"
        m.save(os.path.join(DIR_SALIDA, nombre_archivo))
    
    print(f"✅ Auditoría completada. Abre los archivos HTML generados en {DIR_SALIDA} con tu navegador.")
    print(f"   Ahí podrás ver literalmente las parcelas taladas que la IA detectó.")

if __name__ == "__main__":
    generar_auditoria_satelital()