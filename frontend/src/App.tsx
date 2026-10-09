import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import AnalyticsPanel from './components/AnalyticsPanel';
import type { Feature } from 'geojson';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN as string;
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  
  const [fechas, setFechas] = useState<string[]>([]);
  const [fechaActiva, setFechaActiva] = useState<string>('');
  const [mostrarRegional, setMostrarRegional] = useState(true);
  const [mostrarFisica, setMostrarFisica] = useState(true);
  
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);

  useEffect(() => {
    axios.get(`${API_BASE_URL}/api/fechas`).then(res => {
      setFechas(res.data.fechas);
      if (res.data.fechas.length > 0) setFechaActiva(res.data.fechas[0]);
    }).catch(err => console.error("Error API:", err));
  }, []);

  useEffect(() => {
    if (map.current) return;
    map.current = new mapboxgl.Map({
      container: mapContainer.current!,
      style: 'mapbox://styles/mapbox/satellite-v9',
      center: [-89.5, 19.5],
      zoom: 6.5
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.current.on('load', () => {
      map.current!.addSource('src-regional', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.current!.addLayer({
        id: 'layer-regional',
        type: 'fill',
        source: 'src-regional',
        paint: {
          'fill-color': ['step', ['to-number', ['get', 'PROBABILIDAD_COLAPSO']], '#1a9850', 30, '#ffffbf', 50, '#fdae61', 80, '#d73027'],
          'fill-opacity': ['step', ['to-number', ['get', 'PROBABILIDAD_COLAPSO']], 0.15, 30, 0.4, 50, 0.6, 80, 0.75]
        }
      });

      map.current!.addSource('src-fisica', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      
      map.current!.addLayer({
        id: 'layer-fisica',
        type: 'fill',
        source: 'src-fisica',
        paint: {
          'fill-color': ['case', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], 1], '#ef4444', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], -1], '#22c55e', 'rgba(0,0,0,0)'],
          'fill-opacity': ['case', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], 1], 0.3, ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], -1], 0.3, 0]
        }
      });
      
      map.current!.addLayer({
        id: 'layer-fisica-line',
        type: 'line',
        source: 'src-fisica',
        paint: {
          'line-color': ['case', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], 1], '#ef4444', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], -1], '#22c55e', 'rgba(0,0,0,0)'],
          'line-width': 3
        }
      });

      map.current!.on('click', (e) => {
        const features = map.current!.queryRenderedFeatures(e.point, { layers: ['layer-regional', 'layer-fisica'] });
        if (features.length) setSelectedFeature((features[0] as unknown) as Feature);
        else setSelectedFeature(null);
      });

      ['layer-regional', 'layer-fisica'].forEach(layer => {
        map.current!.on('mouseenter', layer, () => map.current!.getCanvas().style.cursor = 'pointer');
        map.current!.on('mouseleave', layer, () => map.current!.getCanvas().style.cursor = '');
      });
    });
  }, []);

  useEffect(() => {
    // FIX: Encadenamiento opcional según SonarLint (S6582)
    if (!fechaActiva || !map.current?.isStyleLoaded()) return;

    // FIX: Manejo de promesas con .catch según SonarLint (S9383)
    axios.get(`${API_BASE_URL}/api/capas/regional?fecha=${fechaActiva}`)
      .then(res => (map.current?.getSource('src-regional') as mapboxgl.GeoJSONSource)?.setData(res.data))
      .catch(err => console.error("Error sincronizando capa regional:", err));

    axios.get(`${API_BASE_URL}/api/capas/control?fecha=${fechaActiva}`)
      .then(res => (map.current?.getSource('src-fisica') as mapboxgl.GeoJSONSource)?.setData(res.data))
      .catch(err => console.error("Error sincronizando capa de control:", err));
  }, [fechaActiva]);

  useEffect(() => {
    // FIX: Encadenamiento opcional
    if (!map.current?.isStyleLoaded()) return;
    map.current.setLayoutProperty('layer-regional', 'visibility', mostrarRegional ? 'visible' : 'none');
    map.current.setLayoutProperty('layer-fisica', 'visibility', mostrarFisica ? 'visible' : 'none');
    map.current.setLayoutProperty('layer-fisica-line', 'visibility', mostrarFisica ? 'visible' : 'none');
  }, [mostrarRegional, mostrarFisica]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-900 font-sans">
      <Sidebar 
        fechas={fechas} fechaActiva={fechaActiva} setFechaActiva={setFechaActiva}
        mostrarRegional={mostrarRegional} setMostrarRegional={setMostrarRegional}
        mostrarFisica={mostrarFisica} setMostrarFisica={setMostrarFisica}
      />
      <div className="flex-1 relative">
        <div ref={mapContainer} className="absolute inset-0" />
        <AnalyticsPanel feature={selectedFeature} />
      </div>
    </div>
  );
}