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
    }).catch(err => console.error("Error cargando fechas:", err));
  }, []);

  const recargarFechas = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/fechas/recargar`);
      setFechas(res.data.fechas);
      if (res.data.fechas.length > 0) setFechaActiva(res.data.fechas[0]);
    } catch (err) {
      console.error("Error recargando fechas:", err);
    }
  };

  useEffect(() => {
    if (map.current) return;
    
    map.current = new mapboxgl.Map({
      container: mapContainer.current!,
      style: 'mapbox://styles/mapbox/light-v11', 
      center: [-89.5, 19.5],
      zoom: 6.5
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.current.on('load', () => {
      map.current!.resize();
      
      // Capa de Riesgo Regional (Suavizada)
      map.current!.addSource('src-regional', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.current!.addLayer({
        id: 'layer-regional',
        type: 'fill',
        source: 'src-regional',
        paint: {
          'fill-color': ['step', ['to-number', ['get', 'PROBABILIDAD_COLAPSO']], '#22c55e', 30, '#fde047', 50, '#f97316', 80, '#ef4444'],
          'fill-opacity': ['step', ['to-number', ['get', 'PROBABILIDAD_COLAPSO']], 0.02, 30, 0.2, 50, 0.4, 80, 0.6] // Verde casi invisible (0.02)
        }
      });

      // Capa de Zonas de Verificación (Bordes gruesos de alto contraste)
      map.current!.addSource('src-fisica', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.current!.addLayer({
        id: 'layer-fisica',
        type: 'fill',
        source: 'src-fisica',
        paint: {
          'fill-color': 'rgba(0,0,0,0)' // Fondo transparente para ver solo el contorno
        }
      });
      map.current!.addLayer({
        id: 'layer-fisica-line',
        type: 'line',
        source: 'src-fisica',
        paint: {
          'line-color': ['case', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], 1], '#ef4444', ['==', ['to-number', ['get', 'ENSAMBLE_PRED']], -1], '#06b6d4', 'rgba(0,0,0,0)'], // Rojo encendido o Cyan
          'line-width': 4 // Borde más grueso
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
      
      if (fechaActiva) setFechaActiva([...fechaActiva].join('')); 
    });
  }, []);

  useEffect(() => {
    if (!fechaActiva) return;

    const fetchData = async () => {
      try {
        const [resRegional, resControl] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/capas/regional?fecha=${fechaActiva}`),
          axios.get(`${API_BASE_URL}/api/capas/control?fecha=${fechaActiva}`)
        ]);

        const updateMapData = () => {
          const srcReg = map.current?.getSource('src-regional') as mapboxgl.GeoJSONSource;
          const srcFis = map.current?.getSource('src-fisica') as mapboxgl.GeoJSONSource;
          if (srcReg) srcReg.setData(resRegional.data);
          if (srcFis) srcFis.setData(resControl.data);
        };

        if (map.current?.isStyleLoaded()) {
          updateMapData();
        } else {
          map.current?.once('styledata', updateMapData);
        }
      } catch (err) {
        console.error("Error sincronizando capas geográficas:", err);
      }
    };

    fetchData();
  }, [fechaActiva]);

  useEffect(() => {
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
        recargarFechas={recargarFechas} 
      />
      <div className="flex-1 relative h-full">
        <div ref={mapContainer} className="absolute inset-0 h-full w-full" />
      </div>
      <AnalyticsPanel feature={selectedFeature} />
    </div>
  );
}