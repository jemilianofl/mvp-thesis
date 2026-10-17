import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { Feature } from 'geojson';

interface AnalyticsPanelProps {
  readonly feature: Feature | null;
}

export default function AnalyticsPanel({ feature }: AnalyticsPanelProps) {
  if (!feature?.properties) return null;

  // FIX: Reemplazamos 'any' por tipos primitivos estrictos
  const props = feature.properties as Record<string, string | number>;
  
  const rawData = [
    { name: 'NDVI', value: Number(props.NDVI) || 0 },
    { name: 'RADAR VV', value: Number(props.RADAR_VV) || 0 },
    { name: 'RADAR VH', value: Number(props.RADAR_VH) || 0 },
    { name: 'Estrés Hídrico', value: Number(props.ESTRES_HIDRICO) || 0 }
  ];

  const data = rawData.map(d => ({
      ...d,
      fill: d.value < 0 ? '#ef4444' : '#3b82f6'
  }));

  return (
  <div className="w-80 bg-slate-800 p-5 shadow-2xl border-l border-slate-700 z-10 flex flex-col overflow-y-auto flex-shrink-0">
    <h2 className="text-lg font-bold text-white mb-1">Análisis del Tensor</h2>
      <p className="text-sm text-slate-200 mb-4">Polígono / Hex: {props.ID_POLIGONO}</p>

{props.PROBABILIDAD_COLAPSO && (
  <div className="mb-4 bg-slate-900 p-3 rounded border border-slate-700">
    <p className="text-sm text-slate-200">Riesgo Calculado (LSTM)</p>
    <p className="text-2xl font-bold text-emerald-400">{props.PROBABILIDAD_COLAPSO}%</p>
    <p className="text-sm font-medium uppercase mt-1 text-slate-200">{props.NIVEL_RIESGO}</p>
  </div>
)}

      {data.length > 0 ? (
        <div className="h-48 w-full mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 10}} />
              <Tooltip cursor={{fill: '#334155'}} contentStyle={{backgroundColor: '#0f172a', border: 'none', borderRadius: '8px', color: '#fff'}} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="text-xs text-slate-500 italic text-center py-4">Faltan variables físicas para renderizar matriz térmica/satelital.</p>
      )}
    </div>
  );
}