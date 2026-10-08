import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import axios from 'axios';

// FIX: Leer de forma segura desde las variables de entorno de Vercel y forzar el tipado a string
// Leer de forma segura desde las variables de entorno de Vercel
mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN as string;

// Desactivar la telemetría para evitar bloqueos por Ad-Blockers
mapboxgl.workerClass = require('worker-loader!mapbox-gl/dist/mapbox-gl-csp-worker').default;

// URL de Render (Cámbiala cuando Render te asigne un link público)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [fechas, setFechas] = useState<string[]>([]);
  const [fechaActiva, setFechaActiva] = useState<string>('');
  const [mostrarRegional, setMostrarRegional] = useState(true);
  const [mostrarFisica, setMostrarFisica] = useState(true);

  // 1. Cargar Línea de Tiempo al Iniciar
  useEffect(() => {
    axios.get(`${API_BASE_URL}/api/fechas`).then(res => {
      setFechas(res.data.fechas);
      if (res.data.fechas.length > 0) {
        setFechaActiva(res.data.fechas[0]);
      }
    }).catch(err => console.error("Error cargando fechas:", err));
  }, []);

  // 2. Inicializar Mapa Base
  useEffect(() => {
    if (map.current) return; // initialize map only once
    map.current = new mapboxgl.Map({
      container: mapContainer.current!,
      style: 'mapbox://styles/mapbox/satellite-v9',
      center: [-89.5, 19.5],
      zoom: 6.5
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
          'fill-color': ['step', ['get', 'PROBABILIDAD_COLAPSO'], '#1a9850', 30, '#ffffbf', 50, '#fdae61', 80, '#d73027'],
          'fill-opacity': ['step', ['get', 'PROBABILIDAD_COLAPSO'], 0.15, 30, 0.5, 50, 0.7, 80, 0.85],
          'fill-outline-color': 'rgba(255,255,255,0.1)'
        }
      });

      // --- FUENTE 2: VALIDACIÓN FÍSICA (GROUND TRUTH) ---
      map.current!.addSource('src-fisica', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.current!.addLayer({
        id: 'layer-fisica',
        type: 'fill',
        source: 'src-fisica',
        paint: {
          'fill-color': ['match', ['get', 'ENSAMBLE_PRED'], 1, '#ff0000', -1, '#00ff00', 'rgba(0,0,0,0)'],
          'fill-opacity': ['match', ['get', 'ENSAMBLE_PRED'], 1, 0.8, -1, 0.8, 0],
          'fill-outline-color': '#ffffff'
        }
      });

      // INTERACTIVIDAD Y POPUPS (Solución de TypeScript)
      map.current!.on('click', (e) => {
        const features = map.current!.queryRenderedFeatures(e.point);
        if (!features.length) return;

        const featRender = features[0];
        // FIX: Evadir tipado estricto para extraer la capa y propiedades
        const layerId = (featRender as any).layer?.id;
        const props = (featRender as any).properties;

        if (layerId === 'layer-regional') {
          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(`<strong>Riesgo Regional</strong><br>Probabilidad: ${props.PROBABILIDAD_COLAPSO}%<br>Nivel: ${props.NIVEL_RIESGO}`)
            .addTo(map.current!);
        } else if (layerId === 'layer-fisica') {
          const estado = props.ENSAMBLE_PRED === 1 ? 'Degradación Detectada' : 'Biomasa Estable';
          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(`<strong>Auditoría Física</strong><br>Estado: ${estado}<br>Polígono ID: ${props.ID_POLIGONO}`)
            .addTo(map.current!);
        }
      });

      map.current!.on('mouseenter', 'layer-regional', () => { map.current!.getCanvas().style.cursor = 'pointer'; });
      map.current!.on('mouseleave', 'layer-regional', () => { map.current!.getCanvas().style.cursor = ''; });
      map.current!.on('mouseenter', 'layer-fisica', () => { map.current!.getCanvas().style.cursor = 'pointer'; });
      map.current!.on('mouseleave', 'layer-fisica', () => { map.current!.getCanvas().style.cursor = ''; });
    });
  }, []);

  // 3. Sincronizar Capas cuando Cambia la Fecha
  useEffect(() => {
    if (!fechaActiva || !map.current || !map.current.isStyleLoaded()) return;

    // Actualizar Capa Regional
    axios.get(`${API_BASE_URL}/api/capas/regional?fecha=${fechaActiva}`).then(res => {
      const source = map.current?.getSource('src-regional') as mapboxgl.GeoJSONSource;
      if (source) source.setData(res.data);
    }).catch(err => console.error(err));

    // Actualizar Capa Física
    axios.get(`${API_BASE_URL}/api/capas/control?fecha=${fechaActiva}`).then(res => {
      const source = map.current?.getSource('src-fisica') as mapboxgl.GeoJSONSource;
      if (source) source.setData(res.data);
    }).catch(err => console.error(err));

  }, [fechaActiva]);

  // 4. Controladores de Visibilidad
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;
    map.current.setLayoutProperty('layer-regional', 'visibility', mostrarRegional ? 'visible' : 'none');
    map.current.setLayoutProperty('layer-fisica', 'visibility', mostrarFisica ? 'visible' : 'none');
  }, [mostrarRegional, mostrarFisica]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-900 text-slate-100 font-sans">
      
      {/* Panel de Control Lateral */}
      <div className="w-80 bg-slate-800 p-5 flex flex-col gap-6 shadow-2xl z-10 border-r border-slate-700">
        <div>
          <h1 className="text-xl font-bold text-emerald-400">Dashboard MLOps</h1>
          <p className="text-xs text-slate-400 mt-1">Arquitectura Híbrida: LSTM + Ensamble</p>
        </div>

        {/* Máquina del Tiempo */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-semibold text-slate-300">Máquina del Tiempo (Historial):</label>
          <select 
            className="bg-slate-700 border border-slate-600 rounded p-2 text-sm text-white focus:outline-none focus:border-emerald-500"
            value={fechaActiva}
            onChange={(e) => setFechaActiva(e.target.value)}
          >
            {fechas.map(f => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>

        {/* Toggles de Capas */}
        <div className="flex flex-col gap-3 border-t border-slate-700 pt-4">
          <label className="flex items-center gap-3 cursor-pointer">
            <input type="checkbox" checked={mostrarRegional} onChange={(e) => setMostrarRegional(e.target.checked)} className="accent-emerald-500 w-4 h-4"/>
            <span className="text-sm">Capa 1: Predicción Regional (H3)</span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer">
            <input type="checkbox" checked={mostrarFisica} onChange={(e) => setMostrarFisica(e.target.checked)} className="accent-emerald-500 w-4 h-4"/>
            <span className="text-sm">Capa 2: Auditoría Física (Ground Truth)</span>
          </label>
        </div>

        {/* Leyenda Híbrida */}
        <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700 mt-auto">
          <p className="text-xs font-semibold text-slate-400 mb-3">Probabilidad Colapso (IA):</p>
          <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 bg-[#d73027]"></div><span className="text-xs">Crítico (&#62;80%)</span></div>
          <div className="flex items-center gap-2 mb-4"><div className="w-3 h-3 bg-[#fdae61]"></div><span className="text-xs">Alerta Alta (&#62;50%)</span></div>
          
          <p className="text-xs font-semibold text-slate-400 mb-3 border-t border-slate-700 pt-3">Estado Real (Satélite):</p>
          <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 bg-[#ff0000]"></div><span className="text-xs">Degradación Física Detectada</span></div>
          <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 bg-[#00ff00]"></div><span className="text-xs">Biomasa Estable</span></div>
          <div className="flex items-center gap-2"><div className="w-3 h-3 bg-transparent border border-white"></div><span className="text-xs">Nublado / Sin Pase de Sensor</span></div>
        </div>
      </div>

      {/* Contenedor del Mapa */}
      <div ref={mapContainer} className="flex-1 relative" />

    </div>
  );
}

export default App;