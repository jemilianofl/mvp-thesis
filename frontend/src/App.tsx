import { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import axios from 'axios';
import 'mapbox-gl/dist/mapbox-gl.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;

export default function App() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  
  // Estados de la App
  const [hoverInfo, setHoverInfo] = useState<any | null>(null);
  const [fechas, setFechas] = useState<string[]>([]);
  const [fechaActiva, setFechaActiva] = useState<string>('');
  
  // Toggles de Capas
  const [verRegional, setVerRegional] = useState(true);
  const [verControl, setVerControl] = useState(true);

  // 1. Cargar Línea de Tiempo al Iniciar
  useEffect(() => {
    axios.get('http://localhost:8000/api/fechas').then(res => {
      setFechas(res.data.fechas);
      // FIX: Selecciona automáticamente el PRIMER DÍA del dataset real (2026-06-01)
      if (res.data.fechas.length > 0) {
        setFechaActiva(res.data.fechas[0]);
      }
    });
  }, []);

  // 2. Inicializar Mapbox y crear las Fuentes/Capas vacías
  useEffect(() => {
    if (map.current || !mapContainer.current) return;
    
    mapboxgl.accessToken = MAPBOX_TOKEN;
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: [-89.5, 19.5],
      zoom: 6.8,
    });
    
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.current.on('load', () => {
      // --- FUENTE 1: REGIONAL (HEXÁGONOS) ---
      map.current!.addSource('src-regional', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.current!.addLayer({
        id: 'layer-regional',
        type: 'fill',
        source: 'src-regional',
        paint: {
          // FIX: Añadimos un verde sutil (#1a9850) para la Selva Sana (0-30%)
          'fill-color': ['step', ['get', 'PROBABILIDAD_COLAPSO'], '#1a9850', 30, '#ffffbf', 50, '#fdae61', 80, '#d73027'],
          // La selva sana tiene opacidad muy baja (0.15) para no tapar el satélite, las alertas son más sólidas
          'fill-opacity': ['step', ['get', 'PROBABILIDAD_COLAPSO'], 0.15, 30, 0.5, 50, 0.7, 80, 0.85],
          'fill-outline-color': 'rgba(255,255,255,0.1)' // Bordes finos de los hexágonos
        }
      });

      // --- FUENTE 2: GROUND TRUTH (ENSAMBLE) ---
      map.current!.addSource('src-control', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.current!.addLayer({
        id: 'layer-control',
        type: 'fill',
        source: 'src-control',
        paint: {
          'fill-color': [
            'match', ['get', 'ENSAMBLE_PRED'],
            -1, '#d73027',    // Degradación Física = Rojo
             1, '#1a9850',    // Sano = Verde
            'rgba(85,85,85,0.6)' // -999 o default = Gris Translúcido (Nubes)
          ],
          'fill-opacity': 0.85,
          'fill-outline-color': '#ffffff' // Borde blanco grueso para destacarlos sobre H3
        }
      });

      // --- LÓGICA DE HOVER INTELIGENTE ---
      const capasInteractivas = ['layer-regional', 'layer-control'];
      
      map.current!.on('mousemove', (e) => {
        // Consultar qué elementos están debajo del mouse
        const features = map.current!.queryRenderedFeatures(e.point, { layers: capasInteractivas });
        
        if (features.length > 0) {
          // Si el mouse toca ambas capas, priorizamos la capa de Control (Ground Truth)
          const featControl = features.find(f => f.layer.id === 'layer-control');
          const featRender = featControl || features[0];
          
          // Ignorar hover en hexágonos sanos (0% riesgo)
          if (featRender.layer.id === 'layer-regional' && featRender.properties!.PROBABILIDAD_COLAPSO < 30) {
            map.current!.getCanvas().style.cursor = '';
            setHoverInfo(null);
            return;
          }

          map.current!.getCanvas().style.cursor = 'crosshair';
          setHoverInfo({
            x: e.point.x,
            y: e.point.y,
            props: featRender.properties,
            tipo: featRender.layer.id
          });
        } else {
          map.current!.getCanvas().style.cursor = '';
          setHoverInfo(null);
        }
      });
    });
  }, []);

  // 3. Sincronizar Datos cuando cambian la Fecha o los Toggles
  useEffect(() => {
    if (!fechaActiva || !map.current || !map.current.isStyleLoaded()) return;

    // Actualizar Capa Regional (H3)
    if (verRegional) {
      axios.get(`http://localhost:8000/api/capas/regional?fecha=${fechaActiva}`).then(res => {
        (map.current!.getSource('src-regional') as mapboxgl.GeoJSONSource).setData(res.data);
      });
      map.current.setLayoutProperty('layer-regional', 'visibility', 'visible');
    } else {
      map.current.setLayoutProperty('layer-regional', 'visibility', 'none');
    }

    // Actualizar Capa Ground Truth
    if (verControl) {
      axios.get(`http://localhost:8000/api/capas/control?fecha=${fechaActiva}`).then(res => {
        (map.current!.getSource('src-control') as mapboxgl.GeoJSONSource).setData(res.data);
      });
      map.current.setLayoutProperty('layer-control', 'visibility', 'visible');
    } else {
      map.current.setLayoutProperty('layer-control', 'visibility', 'none');
    }
  }, [fechaActiva, verRegional, verControl]);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      
      {/* Panel de Controles (XAI & Time-Series) */}
      <div style={{
        position: 'absolute', top: 20, left: 20, zIndex: 1, 
        background: 'rgba(15,15,15,0.95)', padding: '20px', borderRadius: '8px', 
        color: 'white', fontFamily: 'sans-serif', border: '1px solid #333', width: '320px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.5)'
      }}>
        <h2 style={{ margin: '0 0 5px 0', color: '#4CAF50', fontSize: '20px'}}>Dashboard MLOps</h2>
        <p style={{ fontSize: '13px', color: '#aaa', marginBottom: '20px' }}>Arquitectura Híbrida: LSTM + Ensamble</p>
        
        <label style={{ fontSize: '13px', fontWeight: 'bold', color: '#fff' }}>Máquina del Tiempo (Historial):</label>
        <select 
          value={fechaActiva} 
          onChange={(e) => setFechaActiva(e.target.value)}
          style={{ width: '100%', padding: '8px', marginTop: '8px', marginBottom: '20px', background: '#333', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          {fechas.map(f => <option key={f} value={f}>{f}</option>)}
        </select>

        {/* Toggles */}
        <div style={{ borderTop: '1px solid #444', paddingTop: '15px', marginBottom: '15px', fontSize: '13px' }}>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: '10px', fontWeight: verRegional ? 'bold' : 'normal' }}>
            <input type="checkbox" checked={verRegional} onChange={() => setVerRegional(!verRegional)} style={{ marginRight: '10px' }} />
            Capa 1: Predicción Regional (H3)
          </label>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontWeight: verControl ? 'bold' : 'normal' }}>
            <input type="checkbox" checked={verControl} onChange={() => setVerControl(!verControl)} style={{ marginRight: '10px' }} />
            Capa 2: Auditoría Física (Ground Truth)
          </label>
        </div>

        {/* Leyenda Dual */}
        {verRegional && (
          <div style={{ fontSize: '12px', lineHeight: '2', background: '#222', padding: '10px', borderRadius: '5px', marginBottom:'10px' }}>
            <strong style={{color:'#aaa', display:'block', marginBottom:'5px'}}>Probabilidad Colapso (IA):</strong>
            <div><span style={{display: 'inline-block', width: 12, height: 12, background: '#d73027', marginRight: 8}}></span>Crítico (&gt; 80%)</div>
            <div><span style={{display: 'inline-block', width: 12, height: 12, background: '#fdae61', marginRight: 8}}></span>Alerta Alta (&gt; 50%)</div>
          </div>
        )}
        
        {verControl && (
          <div style={{ fontSize: '12px', lineHeight: '2', background: '#222', padding: '10px', borderRadius: '5px' }}>
            <strong style={{color:'#aaa', display:'block', marginBottom:'5px'}}>Estado Real (Satélite):</strong>
            <div><span style={{display: 'inline-block', width: 12, height: 12, background: '#d73027', border: '1px solid white', marginRight: 8}}></span>Degradación Física Detectada</div>
            <div><span style={{display: 'inline-block', width: 12, height: 12, background: '#1a9850', border: '1px solid white', marginRight: 8}}></span>Biomasa Estable</div>
            <div><span style={{display: 'inline-block', width: 12, height: 12, background: '#555555', border: '1px solid white', marginRight: 8}}></span>Nublado / Sin Pase de Sensor</div>
          </div>
        )}
      </div>

      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

      {/* Tooltip Dinámico */}
      {hoverInfo && (
        <div style={{
          position: 'absolute', left: hoverInfo.x + 20, top: hoverInfo.y - 20,
          background: 'rgba(255, 255, 255, 0.95)', color: '#222', padding: '15px', borderRadius: '6px',
          pointerEvents: 'none', fontFamily: 'sans-serif', boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          zIndex: 2, minWidth: '220px'
        }}>
          {hoverInfo.tipo === 'layer-regional' ? (
            <>
              <h4 style={{ margin: '0 0 10px 0', borderBottom: '1px solid #ccc', paddingBottom: '5px', color: '#8b0000' }}>Hexágono H3: {hoverInfo.props.ID_POLIGONO}</h4>
              <strong>Predicción LSTM:</strong> {hoverInfo.props.NIVEL_RIESGO}<br/>
              <strong>Riesgo de Pérdida:</strong> {hoverInfo.props.PROBABILIDAD_COLAPSO}%
            </>
          ) : (
            <>
              <h4 style={{ margin: '0 0 10px 0', borderBottom: '1px solid #ccc', paddingBottom: '5px', color: '#105b9b' }}>Parcela Maestra #{hoverInfo.props.ID_POLIGONO}</h4>
              <strong>Estado en Terreno:</strong> <span style={{fontWeight:'bold', color: hoverInfo.props.ENSAMBLE_PRED === -1 ? '#d73027' : '#1a9850'}}>{hoverInfo.props.ENSAMBLE_PRED === -1 ? 'Degradación' : hoverInfo.props.ENSAMBLE_PRED === 1 ? 'Sano' : 'Nubes (Sin Datos)'}</span><br/>
              <hr style={{ border: '0.5px solid #eee', margin: '8px 0' }}/>
              <span style={{fontSize:'11px', color:'#666'}}>Lecturas de Teledetección ({fechaActiva}):</span><br/>
              <strong>Vigor (NDVI):</strong> {hoverInfo.props.NDVI}<br/>
              <strong>Dosel (Radar VV):</strong> {hoverInfo.props.RADAR_VV}<br/>
              <strong>Estrés Hídrico:</strong> {hoverInfo.props.ESTRES_HIDRICO}
            </>
          )}
        </div>
      )}
    </div>
  );
}