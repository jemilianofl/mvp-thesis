import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import axios from 'axios';
import type { Feature } from 'geojson';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface AnalyticsPanelProps {
  readonly feature: Feature | null;
}

export default function AnalyticsPanel({ feature }: AnalyticsPanelProps) {
  const [timeSeries, setTimeSeries] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const idPoligono = feature?.properties?.ID_POLIGONO;
    if (!idPoligono) {
      setTimeSeries([]);
      return;
    }
    
    setLoading(true);
    axios.get(`${API_BASE_URL}/api/serie_tiempo/${idPoligono}`)
      .then(res => setTimeSeries(res.data))
      .catch(err => console.error("Error cargando serie temporal:", err))
      .finally(() => setLoading(false));
  }, [feature]);

  if (!feature?.properties) return null;
  const props = feature.properties as Record<string, string | number>;
  
  const lastDataPoint = timeSeries.length > 0 ? timeSeries[timeSeries.length - 1] : {};

  return (
    <div className="w-[450px] bg-slate-800 p-5 shadow-2xl border-l border-slate-700 z-10 flex flex-col overflow-y-auto flex-shrink-0">
      <h2 className="text-lg font-bold text-white mb-1">Evolución del Cuadrante</h2>
      <p className="text-sm text-slate-400 mb-4">Identificador: {props.ID_POLIGONO}</p>

      {props.PROBABILIDAD_COLAPSO !== undefined && (
        <div className="mb-4 bg-slate-900 p-3 rounded border border-slate-700 flex justify-between items-center shadow-inner">
          <div>
            <p className="text-xs text-slate-400">Riesgo Calculado</p>
            <p className="text-2xl font-bold text-emerald-400">{props.PROBABILIDAD_COLAPSO}%</p>
          </div>
          <div className="text-right">
             <p className="text-xs text-slate-400">Diagnóstico Preventivo</p>
             <p className="text-sm font-bold uppercase mt-1 text-slate-200">{props.NIVEL_RIESGO || 'EN ANÁLISIS'}</p>
          </div>
        </div>
      )}

      <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-2">Comportamiento Histórico Base</h3>
      <div className="min-h-[220px] w-full">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-slate-500 text-sm animate-pulse">Consultando lecturas satelitales...</p>
          </div>
        ) : timeSeries.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeries} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="fecha" stroke="#94a3b8" fontSize={10} tickMargin={10} />
              
              <YAxis yAxisId="left" stroke="#94a3b8" fontSize={10} domain={['auto', 'auto']} />
              <YAxis yAxisId="right" orientation="right" stroke="#fbbf24" fontSize={10} domain={[0, 100]} />
              
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#fff' }}
                itemStyle={{ fontSize: '12px' }}
                labelStyle={{ fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              
              <Line yAxisId="left" type="monotone" dataKey="NDVI_Graph" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} name="Índice de Vegetación (NDVI)" />
              <Line yAxisId="left" type="monotone" dataKey="Estres_Graph" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} name="Estrés Hídrico" />
              <Line yAxisId="right" type="monotone" dataKey="Riesgo_Graph" stroke="#fbbf24" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 4 }} activeDot={{ r: 6 }} name="Probabilidad Riesgo (%)" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-slate-500 italic text-center py-10">Sin registro histórico para trazar evolución.</p>
        )}
      </div>
      
      <h3 className="text-sm font-semibold text-slate-200 mt-6 mb-2">Lectura Forestal Detallada</h3>
      <div className="grid grid-cols-3 gap-2">
        {Object.entries(lastDataPoint).map(([key, value]) => {
          if (['fecha', 'NDVI_Graph', 'Estres_Graph', 'Riesgo_Graph', 'PROBABILIDAD_COLAPSO', 'ENSAMBLE_PRED'].includes(key)) return null;
          
          return (
            <div key={key} className="bg-slate-900 p-2 rounded border border-slate-700 shadow flex flex-col justify-center items-center text-center">
              <p className="text-[9px] text-slate-400 truncate w-full uppercase tracking-wider">{key.replace(/_/g, ' ')}</p>
              <p className="text-sm font-semibold text-slate-200 mt-1">{Number(value).toFixed(2)}</p>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-slate-500 mt-6 leading-relaxed border-t border-slate-700 pt-4">
        * Nota Forestal: Esta matriz refleja las lecturas biométricas y físicas del suelo. La correlación entre la caída de vegetación y las anomalías de terreno permiten pronosticar la degradación (línea amarilla).
      </p>
    </div>
  );
}