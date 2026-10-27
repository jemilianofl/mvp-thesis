import { useEffect, useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import axios from 'axios';
import type { Feature } from 'geojson';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const COLORS = ['#10b981', '#3b82f6', '#ec4899', '#8b5cf6', '#06b6d4', '#eab308', '#a855f7', '#14b8a6', '#f43f5e', '#84cc16', '#6366f1', '#f97316'];

interface AnalyticsPanelProps {
  readonly feature: Feature | null;
  readonly fecha: string;
  readonly onClose: () => void;
}

export default function AnalyticsPanel({ feature, fecha, onClose }: AnalyticsPanelProps) {
  const [timeSeries, setTimeSeries] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const idPoligono = feature?.properties?.ID_POLIGONO;
    if (!idPoligono) {
      setTimeSeries([]);
      return;
    }
    
    setLoading(true);
    setError(null);
    axios.get(`${API_BASE_URL}/api/serie_tiempo/${idPoligono}?fecha=${fecha}`)
      .then(res => setTimeSeries(res.data))
      .catch(err => {
        console.error("Error cargando serie temporal:", err);
        setError("Fallo al consultar historial del cuadrante.");
      })
      .finally(() => setLoading(false));
  }, [feature, fecha]);

  const dynamicKeys = useMemo(() => {
    if (timeSeries.length === 0) return [];
    // Filtramos las llaves que usa el sistema internamente
    return Object.keys(timeSeries[0]).filter(key => 
      !['fecha', 'Riesgo_IA', 'PROBABILIDAD_COLAPSO', 'ENSAMBLE_PRED', 'ID_POLIGONO', 'NIVEL_RIESGO'].includes(key)
    );
  }, [timeSeries]);

  if (!feature || !feature.properties) {
      return (
          <div className="w-[450px] bg-slate-800 p-5 shadow-2xl border-l border-slate-700 z-10 flex flex-col justify-center items-center flex-shrink-0">
             <p className="text-slate-500 text-sm">Seleccione un cuadrante en el mapa para ver su evolución.</p>
          </div>
      );
  }
  
  const props = feature.properties as Record<string, any>;
  const lastDataPoint = timeSeries.length > 0 ? timeSeries[timeSeries.length - 1] : {};

  const riesgoNum = Number(props.PROBABILIDAD_COLAPSO) || 0;
  const nivelRiesgoSeguro = String(props.NIVEL_RIESGO || (riesgoNum > 0 ? 'CALCULADO' : 'ESTABLE')).toUpperCase();
  const hexId = String(props.ID_POLIGONO || 'N/A');
  
  // Validación de Ground Truth (Auditoría Física)
  const esAnomaliaFisica = Number(props.ENSAMBLE_PRED) === 1;

  return (
    <div className="w-[450px] bg-slate-800 p-5 shadow-2xl border-l border-slate-700 z-10 flex flex-col overflow-y-auto flex-shrink-0 relative">
      
      <button 
        onClick={onClose} 
        className="absolute top-4 right-4 text-slate-400 hover:text-white font-bold text-lg cursor-pointer"
        title="Cerrar panel"
      >
        ✕
      </button>

      <h2 className="text-lg font-bold text-white mb-1 pr-6">Evolución del Cuadrante</h2>
      <p className="text-sm text-slate-400 mb-4">Identificador: {hexId}</p>

      <div className="mb-4 bg-slate-900 p-3 rounded border border-slate-700 flex justify-between items-center shadow-inner">
        <div>
          <p className="text-xs text-slate-400">Riesgo Calculado (IA)</p>
          <p className={`text-2xl font-bold ${riesgoNum > 50 ? 'text-red-400' : 'text-emerald-400'}`}>{riesgoNum}%</p>
        </div>
        <div className="text-right">
            <p className="text-xs text-slate-400">Diagnóstico Preventivo</p>
            <p className="text-sm font-bold uppercase mt-1 text-slate-200">{nivelRiesgoSeguro}</p>
        </div>
      </div>
      
      {/* Etiqueta de Validación Física */}
      {esAnomaliaFisica && (
          <div className="mb-4 bg-red-900/30 border border-red-500/50 p-2 rounded text-center">
              <span className="text-xs text-red-400 font-bold uppercase tracking-wider">⚠️ Degradación Física Confirmada por Satélite</span>
          </div>
      )}

      <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-2">Comportamiento Semanal (Últimos 7 Días)</h3>
      <div className="min-h-[300px] w-full">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-slate-500 text-sm animate-pulse">Consultando lecturas satelitales...</p>
          </div>
        ) : error ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        ) : timeSeries.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeries} margin={{ top: 10, right: 0, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              
              <XAxis dataKey="fecha" stroke="#94a3b8" fontSize={10} tickMargin={5} minTickGap={20} angle={-45} textAnchor="end" />
              <YAxis yAxisId="left" stroke="#94a3b8" fontSize={10} domain={['auto', 'auto']} />
              <YAxis yAxisId="right" orientation="right" stroke="#fbbf24" fontSize={10} domain={[0, 100]} />
              
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#fff' }}
                itemStyle={{ fontSize: '12px' }}
                labelStyle={{ fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '20px' }} />
              
              {dynamicKeys.map((key, index) => (
                <Line 
                  key={key}
                  yAxisId="left" 
                  type="monotone" 
                  dataKey={key} 
                  stroke={COLORS[index % COLORS.length]} 
                  strokeWidth={2} 
                  dot={{ r: 3 }} 
                  activeDot={{ r: 5 }} 
                  name={key.replace(/_/g, ' ')} 
                />
              ))}
              
              <Line 
                yAxisId="right" 
                type="stepAfter" 
                dataKey="Riesgo_IA" 
                stroke="#fbbf24" 
                strokeWidth={2} 
                strokeDasharray="4 4" 
                dot={{ r: 3 }} 
                activeDot={{ r: 5 }} 
                name="Probabilidad Riesgo (%)" 
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-slate-500 italic text-center py-10">Sin registro histórico para trazar evolución.</p>
        )}
      </div>
      
      <h3 className="text-sm font-semibold text-slate-200 mt-6 mb-2">Matriz de Variables (Última Lectura)</h3>
      <div className="grid grid-cols-3 gap-2">
        {Object.entries(lastDataPoint).map(([key, value]) => {
          if (['fecha', 'Riesgo_IA', 'PROBABILIDAD_COLAPSO', 'ENSAMBLE_PRED'].includes(key)) return null;
          return (
            <div key={key} className="bg-slate-900 p-2 rounded border border-slate-700 shadow flex flex-col justify-center items-center text-center">
              <p className="text-[9px] text-slate-400 truncate w-full uppercase tracking-wider" title={key.replace(/_/g, ' ')}>
                  {key.replace(/_/g, ' ')}
              </p>
              <p className="text-sm font-semibold text-slate-200 mt-1">{Number(value).toFixed(2)}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}