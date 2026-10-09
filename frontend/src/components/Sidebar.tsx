// FIX: Tipado estricto y seguro en lugar de 'any'
interface SidebarProps {
  readonly fechas: string[];
  readonly fechaActiva: string;
  readonly setFechaActiva: (f: string) => void;
  readonly mostrarRegional: boolean;
  readonly setMostrarRegional: (v: boolean) => void;
  readonly mostrarFisica: boolean;
  readonly setMostrarFisica: (v: boolean) => void;
}

export default function Sidebar({ 
  fechas, 
  fechaActiva, 
  setFechaActiva, 
  mostrarRegional, 
  setMostrarRegional, 
  mostrarFisica, 
  setMostrarFisica 
}: SidebarProps) {
  return (
    <div className="w-80 bg-slate-800 p-5 flex flex-col gap-6 shadow-2xl z-10 border-r border-slate-700 overflow-y-auto">
      <div>
        <h1 className="text-xl font-bold text-emerald-400">Dashboard MLOps</h1>
        <p className="text-xs text-slate-400 mt-1">Arquitectura Híbrida: LSTM + Ensamble</p>
      </div>

      <div className="flex flex-col gap-2">
        {/* FIX: Se añade htmlFor y un ID en el select para accesibilidad HTML (S6853) */}
        <label htmlFor="selector-fecha" className="text-sm font-semibold text-slate-300">Máquina del Tiempo:</label>
        <select 
          id="selector-fecha"
          className="bg-slate-700 border border-slate-600 rounded p-2 text-sm text-white focus:border-emerald-500 outline-none"
          value={fechaActiva}
          onChange={(e) => setFechaActiva(e.target.value)}
        >
          {fechas.map((f: string) => <option key={f} value={f}>{f}</option>)}
        </select>
      </div>

      <div className="flex flex-col gap-3 border-t border-slate-700 pt-4">
        <label className="flex items-center gap-3 cursor-pointer">
          <input type="checkbox" checked={mostrarRegional} onChange={(e) => setMostrarRegional(e.target.checked)} className="accent-emerald-500 w-4 h-4"/>
          <span className="text-sm font-medium">Capa Regional (H3)</span>
        </label>
        <label className="flex items-center gap-3 cursor-pointer">
          <input type="checkbox" checked={mostrarFisica} onChange={(e) => setMostrarFisica(e.target.checked)} className="accent-emerald-500 w-4 h-4"/>
          <span className="text-sm font-medium">Auditoría Física (Ground Truth)</span>
        </label>
      </div>

      <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700 mt-auto">
        <p className="text-xs font-semibold text-slate-400 mb-3">Probabilidad Colapso (IA):</p>
        <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 bg-[#d73027]"></div><span className="text-xs">Crítico (&#62;80%)</span></div>
        <div className="flex items-center gap-2 mb-4"><div className="w-3 h-3 bg-[#fdae61]"></div><span className="text-xs">Alerta Alta (&#62;50%)</span></div>
        <p className="text-xs font-semibold text-slate-400 mb-3 border-t border-slate-700 pt-3">Estado Real (Satélite):</p>
        <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 border-[2px] border-[#ef4444] bg-[#ef4444]/30"></div><span className="text-xs">Degradación Detectada</span></div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 border-[2px] border-[#22c55e] bg-[#22c55e]/30"></div><span className="text-xs">Biomasa Estable</span></div>
      </div>
    </div>
  );
}