import type { FechaObj } from '../App';

interface SidebarProps {
  readonly fechas: FechaObj[];
  readonly fechaActiva: string;
  readonly setFechaActiva: (f: string) => void;
  readonly mostrarRegional: boolean;
  readonly setMostrarRegional: (v: boolean) => void;
  readonly mostrarFisica: boolean;
  readonly setMostrarFisica: (v: boolean) => void;
  readonly recargarFechas: () => void; 
}

export default function Sidebar({ 
  fechas, fechaActiva, setFechaActiva, 
  mostrarRegional, setMostrarRegional, 
  mostrarFisica, setMostrarFisica, recargarFechas 
}: SidebarProps) {
  return (
    <div className="w-80 bg-slate-800 p-5 flex flex-col gap-6 shadow-2xl z-10 border-r border-slate-700 overflow-y-auto">
      <div>
        <h1 className="text-xl font-bold text-emerald-400">Monitoreo Forestal</h1>
        <p className="text-sm text-slate-200 mt-1">Sistema de Alerta Temprana de Deforestación</p>
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="selector-fecha" className="text-sm font-semibold text-slate-200">Fecha de Observación:</label>
        <select 
          id="selector-fecha"
          className="bg-slate-700 border border-slate-600 rounded p-2 text-sm text-white focus:border-emerald-500 outline-none cursor-pointer"
          value={fechaActiva}
          onChange={(e) => setFechaActiva(e.target.value)}
        >
          {fechas.map((f: FechaObj) => (
            <option key={f.fecha} value={f.fecha}>
              {f.alerta ? `⚠️ ${f.fecha} (DÍA DE ALERTA)` : `✅ ${f.fecha}`}
            </option>
          ))}
        </select>
        <button 
          onClick={recargarFechas}
          className="mt-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold py-2 px-4 rounded transition-colors"
        >
          ↻ Actualizar Datos Satelitales
        </button>
      </div>

      <div className="flex flex-col gap-3 border-t border-slate-700 pt-4">
        <label className="flex items-center gap-3 cursor-pointer">
          <input type="checkbox" checked={mostrarRegional} onChange={(e) => setMostrarRegional(e.target.checked)} className="accent-emerald-500 w-4 h-4"/>
          <span className="text-sm font-medium text-slate-200">Mapa de Riesgo Regional</span>
        </label>
        <label className="flex items-center gap-3 cursor-pointer">
          <input type="checkbox" checked={mostrarFisica} onChange={(e) => setMostrarFisica(e.target.checked)} className="accent-emerald-500 w-4 h-4"/>
          <span className="text-sm font-medium text-slate-200">Zonas de Verificación en Campo</span>
        </label>
      </div>

      <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700 mt-auto">
        <p className="text-sm font-semibold text-slate-200 mb-3">Pronóstico de Degradación:</p>
        <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 bg-[#ef4444]"></div><span className="text-sm text-slate-200">Crítico (&#62;80%)</span></div>
        <div className="flex items-center gap-2 mb-4"><div className="w-3 h-3 bg-[#f97316]"></div><span className="text-sm text-slate-200">Alerta Alta (&#62;50%)</span></div>
        
        <p className="text-sm font-semibold text-slate-200 mb-3 border-t border-slate-700 pt-3">Detección Histórica (Satélite):</p>
        <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 border-[2px] border-[#ef4444] bg-[#ef4444]/30"></div><span className="text-sm text-slate-200">Deforestación Confirmada</span></div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 border-[2px] border-[#06b6d4] bg-[#06b6d4]/30"></div><span className="text-sm text-slate-200">Biomasa Intacta</span></div>
      </div>
    </div>
  );
}