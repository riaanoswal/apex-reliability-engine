import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  Cpu,
  Download,
  ShieldCheck,
  Zap,
  Terminal,
  Server,
  Cloud,
  Layers,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Send,
  Radio,
  Clock,
  Settings
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid
} from 'recharts';

// Empty string routes /api/* through the Vite dev-server proxy (vite.config.ts),
// so the dashboard keeps working when the backend is moved to another port.
// Set VITE_API_BASE to call a backend directly instead.
const API_BASE = import.meta.env.VITE_API_BASE ?? '';

interface TelemetryPoint {
  time_ms: number;
  amplitude_g: number;
}

interface SpectrumPoint {
  frequency_hz: number;
  psd: number;
}

interface TelemetryData {
  asset_id: string;
  rms_velocity_mms: number;
  crest_factor: number;
  spectral_kurtosis: number;
  bpfo_frequency_hz: number;
  iso_10816_status: string;
  iso_zone: 'A' | 'B' | 'C' | 'D';
  machine_class: string;
  diagnosis: string;
  shaft_rpm: number;
  envelope_scores_db: { BPFO: number; BPFI: number; BSF: number };
  defect_frequencies_hz: { BPFO: number; BPFI: number; BSF: number; FTF: number };
  time_series: TelemetryPoint[];
  fft_spectrum: SpectrumPoint[];
}

interface LatencyTrace {
  cusignal_dsp_ms: number;
  cuvs_rag_retrieval_ms: number;
  nemo_guardrails_ms: number;
  llm_ttft_ms: number;
  total_pipeline_ms: number;
  hardware_speedup_factor: number;
}

interface DiagnoseResponse {
  asset_id: string;
  mode: string;
  path: string;
  model_used: string;
  diagnosis: string;
  diagnostic_report: string;
  matched_cmms_record: string | null;
  similarity_score: number;
  risk_priority_number: number;
  rpn_severity: number;
  rpn_occurrence: number;
  rpn_detection: number;
  priority: string;
  action_priority: string;
  parts_allocated: string[];
  loto_protocol: string;
  checks: Record<string, boolean>;
  execution_trace: LatencyTrace;
}

// ISO 10816-3 zone -> card styling. Full class strings so Tailwind keeps them.
const ZONE_STYLE: Record<string, { border: string; bar: string; badgeBg: string; text: string; note: string }> = {
  A: { border: 'border-emerald-500/40', bar: 'bg-emerald-500', badgeBg: 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40', text: 'text-emerald-400/80', note: 'Newly commissioned range' },
  B: { border: 'border-emerald-500/40', bar: 'bg-emerald-500', badgeBg: 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40', text: 'text-emerald-400/80', note: 'Acceptable for long-term operation' },
  C: { border: 'border-amber-500/40', bar: 'bg-amber-500', badgeBg: 'bg-amber-950/80 text-amber-300 border-amber-500/40', text: 'text-amber-400/80', note: 'Unsatisfactory for continuous operation' },
  D: { border: 'border-rose-500/40', bar: 'bg-rose-500', badgeBg: 'bg-rose-950/80 text-rose-400 border-rose-500/50', text: 'text-rose-400/80', note: 'Severe - damage may occur' }
};

const FAULT_LABEL: Record<string, string> = {
  OR: 'Outer-race defect (BPFO)',
  IR: 'Inner-race defect (BPFI)',
  Ball: 'Rolling-element defect (BSF)',
  Normal: 'No bearing defect signature'
};

export default function App() {
  const [mode, setMode] = useState<'cloud' | 'local'>('cloud');
  const [apiKey, setApiKey] = useState<string>('');
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [loadingTelemetry, setLoadingTelemetry] = useState<boolean>(true);

  const [query, setQuery] = useState<string>('Assess the BPFO harmonic peak and confirm bearing replacement protocol');
  const [diagnosis, setDiagnosis] = useState<DiagnoseResponse | null>(null);
  const [loadingDiagnosis, setLoadingDiagnosis] = useState<boolean>(false);
  const [traceOpen, setTraceOpen] = useState<boolean>(true);

  // Fetch telemetry from FastAPI backend
  const fetchTelemetry = async () => {
    try {
      setLoadingTelemetry(true);
      const res = await fetch(`${API_BASE}/api/telemetry`);
      if (!res.ok) throw new Error('Failed to retrieve telemetry stream');
      const data: TelemetryData = await res.json();
      setTelemetry(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingTelemetry(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
  }, []);

  const handleDiagnose = async (customQuery?: string) => {
    const q = customQuery || query;
    try {
      setLoadingDiagnosis(true);
      const res = await fetch(`${API_BASE}/api/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          mode: mode,
          api_key: apiKey || undefined
        })
      });
      if (!res.ok) throw new Error('Diagnostic query failed');
      const data: DiagnoseResponse = await res.json();
      setDiagnosis(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDiagnosis(false);
    }
  };

  // Download ISO-14224 SAP PM Work Order Payload
  const handleDownloadWorkOrder = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/work-order`);
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `WORK_ORDER_ISO14224_${data.work_order_id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Work order generation failed', err);
    }
  };

  const zone = telemetry ? telemetry.iso_zone : 'A';
  const zs = ZONE_STYLE[zone] || ZONE_STYLE.A;
  const bpfo = telemetry ? telemetry.bpfo_frequency_hz : 105.9;
  const checksPassed = diagnosis ? Object.values(diagnosis.checks).filter(Boolean).length : 0;
  const checksTotal = diagnosis ? Object.values(diagnosis.checks).length : 4;

  return (
    <div className="min-h-screen bg-[#06090e] text-slate-200 scada-grid-bg flex flex-col font-sans">
      {/* 1. SCADA HEADER CONTROLLER BAR */}
      <header className="h-16 border-b border-slate-800/80 bg-[#090d16]/90 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="h-7 w-7 rounded bg-gradient-to-tr from-cyan-600 to-emerald-400 flex items-center justify-center font-mono font-black text-black text-sm">
              ▲
            </div>
            <span className="font-mono text-base font-bold tracking-wider text-white">
              APEX<span className="text-cyan-400 font-light ml-1.5 text-xs px-1.5 py-0.5 border border-cyan-500/30 rounded bg-cyan-950/40">PS137</span>
            </span>
          </div>
          <div className="h-4 w-[1px] bg-slate-800" />
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className="text-slate-600">ISA-95:</span>
            <span className="text-slate-200">GULF_FPSO_01</span>
            <span className="text-slate-600">/</span>
            <span className="text-slate-200">TRAIN_02</span>
            <span className="text-slate-600">/</span>
            <span className="text-cyan-400 bg-cyan-950/40 px-1.5 py-0.5 rounded border border-cyan-500/20">CMP-204B</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Heartbeat Status Indicator */}
          <div className="flex items-center gap-2 bg-slate-900/80 border border-slate-800 px-3 py-1 rounded text-xs font-mono">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-emerald-400 font-medium">CWRU REPLAY (12 kHz)</span>
          </div>

          {/* Mode Selector Toggle: Cloud vs Air-Gapped Edge */}
          <div className="flex items-center bg-slate-950 p-1 border border-slate-800 rounded">
            <button
              onClick={() => setMode('cloud')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono transition-all ${
                mode === 'cloud'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Cloud className="w-3.5 h-3.5" />
              Cloud (Gemini)
            </button>
            <button
              onClick={() => setMode('local')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-mono transition-all ${
                mode === 'local'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              Edge (Air-Gapped)
            </button>
          </div>

          {mode === 'cloud' && (
            <input
              type="password"
              placeholder="Google AI Studio Key..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="bg-slate-950 border border-slate-800 text-xs px-2.5 py-1.5 rounded font-mono text-slate-300 w-44 focus:outline-none focus:border-cyan-500"
            />
          )}

          <button
            onClick={fetchTelemetry}
            className="p-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded text-slate-400 hover:text-slate-200 transition"
            title="Refresh SCADA Ingestion"
          >
            <RefreshCw className={`w-4 h-4 ${loadingTelemetry ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 space-y-6 max-w-[1750px] mx-auto w-full">
        {/* 2. TOP INDUSTRIAL KPI MATRIX */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className={`bg-[#0b101b] border ${zs.border} rounded p-4 relative overflow-hidden`}>
            <div className={`absolute top-0 right-0 h-1 w-full ${zs.bar}`} />
            <div className="flex justify-between items-start">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Vibration RMS</span>
              <span className={`flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded border ${zs.badgeBg}`}>
                ZONE {zone}
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-mono font-bold text-white tracking-tight">
                {telemetry ? telemetry.rms_velocity_mms.toFixed(2) : '—'}
              </span>
              <span className="text-xs font-mono text-slate-500">mm/s</span>
            </div>
            <p className={`mt-1 text-[11px] font-mono ${zs.text}`}>ISO 10816-3, 10–1000 Hz</p>
          </div>

          <div className="bg-[#0b101b] border border-amber-500/40 rounded p-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 h-1 w-full bg-amber-500" />
            <div className="flex justify-between items-start">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Spectral Kurtosis</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-500/40">
                {telemetry && telemetry.spectral_kurtosis > 3 ? 'k > 3.00' : 'k ≈ 3.00'}
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-mono font-bold text-white tracking-tight">
                {telemetry ? telemetry.spectral_kurtosis.toFixed(2) : '—'}
              </span>
              <span className="text-xs font-mono text-slate-500">4th Mom.</span>
            </div>
            <p className="mt-1 text-[11px] font-mono text-amber-400/80">
              {telemetry ? `Crest factor ${telemetry.crest_factor.toFixed(2)}` : 'Impulsiveness indicator'}
            </p>
          </div>

          <div className="bg-[#0b101b] border border-cyan-500/40 rounded p-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 h-1 w-full bg-cyan-500" />
            <div className="flex justify-between items-start">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Dynamic RPN</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-500/40">
                {diagnosis ? diagnosis.priority : 'AWAITING RUN'}
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-mono font-bold text-white tracking-tight">
                {diagnosis ? diagnosis.risk_priority_number : '—'}
              </span>
              <span className="text-xs font-mono text-slate-500">/ 1000</span>
            </div>
            <p className="mt-1 text-[11px] font-mono text-cyan-400/80">
              {diagnosis
                ? `FMEA: S(${diagnosis.rpn_severity}) × O(${diagnosis.rpn_occurrence}) × D(${diagnosis.rpn_detection})`
                : 'FMEA: S × O × D'}
            </p>
          </div>

          <div className="bg-[#0b101b] border border-slate-800 rounded p-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 h-1 w-full bg-emerald-500" />
            <div className="flex justify-between items-start">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Loop Latency</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/40">
                MEASURED
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-mono font-bold text-white tracking-tight">
                {diagnosis ? diagnosis.execution_trace.total_pipeline_ms : '—'}
              </span>
              <span className="text-xs font-mono text-slate-500">ms</span>
            </div>
            <p className="mt-1 text-[11px] font-mono text-emerald-400/80">NumPy / SciPy on CPU</p>
          </div>

          <div className="bg-[#0b101b] border border-slate-800 rounded p-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 h-1 w-full bg-indigo-500" />
            <div className="flex justify-between items-start">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Output Checks</span>
              <ShieldCheck className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-mono font-bold text-white tracking-tight">
                {diagnosis ? `${checksPassed}/${checksTotal}` : '—'}
              </span>
              <span className="text-xs font-mono text-indigo-400">
                {diagnosis ? (checksPassed === checksTotal ? 'PASSED' : 'FAILED') : 'PENDING'}
              </span>
            </div>
            <p className="mt-1 text-[11px] font-mono text-slate-400">Grounding · Sections · IDs · Safety</p>
          </div>
        </div>

        {/* 3. CORE SPLIT SCREEN: TELEMETRY DSP vs RELIABILITY COPILOT */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* LEFT: WAVEFORM & FFT SPECTRAL DENSITY */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-[#090d16] border border-slate-800 rounded p-5">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs font-mono uppercase tracking-wider font-semibold text-slate-200">
                    Drive-End Vibration Acceleration Waveform (12 kHz CWRU)
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-slate-500">Axis: Radial DE</span>
              </div>
              <div className="h-52 w-full">
                {telemetry ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={telemetry.time_series} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="time_ms" stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} unit="ms" />
                      <YAxis stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} unit="g" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }}
                        labelStyle={{ color: '#94a3b8' }}
                      />
                      <Line type="monotone" dataKey="amplitude_g" stroke="#06b6d4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs font-mono text-slate-600">Ingesting Telemetry Buffer...</div>
                )}
              </div>
            </div>

            <div className="bg-[#090d16] border border-slate-800 rounded p-5">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <Radio className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-xs font-mono uppercase tracking-wider font-semibold text-slate-200">
                    Envelope Spectrum (Hilbert) — Peak Kinematic Harmonics
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-rose-400 bg-rose-950/40 border border-rose-800/40 px-2 py-0.5 rounded">
                  Target BPFO: {bpfo.toFixed(1)} Hz
                </span>
              </div>
              <div className="h-56 w-full">
                {telemetry ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={telemetry.fft_spectrum} margin={{ top: 22, right: 20, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="frequency_hz" stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} unit="Hz" />
                      <YAxis stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }}
                      />
                      <ReferenceLine
                        x={Math.round(bpfo)}
                        stroke="#f43f5e"
                        strokeDasharray="4 4"
                        strokeWidth={2}
                        label={{ value: `1X BPFO (${bpfo.toFixed(1)} Hz)`, position: 'top', fill: '#f43f5e', fontSize: 11, fontFamily: 'monospace' }}
                      />
                      <Line type="monotone" dataKey="psd" stroke="#10b981" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs font-mono text-slate-600">Executing Envelope Demodulation...</div>
                )}
              </div>
              <p className="mt-2 text-[10px] font-mono text-slate-600">
                Hilbert envelope of the spectral-kurtosis band, 1 Hz bins (mg). Peaks at BPFO and its harmonics confirm an outer-race defect.
              </p>
            </div>
          </div>

          {/* RIGHT: AUTONOMOUS COPILOT TERMINAL */}
          <div className="lg:col-span-5 flex flex-col space-y-4">
            <div className="bg-[#090d16] border border-slate-800 rounded p-5 flex-1 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-center pb-3 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-mono uppercase tracking-wider font-semibold text-slate-200">
                      Reliability Multi-Agent Terminal
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                    Engine: {diagnosis ? diagnosis.model_used : mode === 'cloud' ? 'Gemini (cloud)' : 'Ollama (local)'}
                  </span>
                </div>

                <div className="flex flex-wrap gap-2 my-3">
                  <button
                    onClick={() => {
                      const q = 'Execute ISO 14224 root cause failure analysis on the detected bearing defect frequency';
                      setQuery(q);
                      handleDiagnose(q);
                    }}
                    className="text-[10px] font-mono bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/60 px-2 py-1 rounded transition"
                  >
                    [RCA BPFO Trip]
                  </button>
                  <button
                    onClick={() => {
                      const q = 'Verify safety lockout isolation key and bill of materials';
                      setQuery(q);
                      handleDiagnose(q);
                    }}
                    className="text-[10px] font-mono bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/60 px-2 py-1 rounded transition"
                  >
                    [Verify LOTO & Parts]
                  </button>
                  <button
                    onClick={() => {
                      const q = 'Calculate catastrophic runaway MTBF and stage priority work order';
                      setQuery(q);
                      handleDiagnose(q);
                    }}
                    className="text-[10px] font-mono bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700/60 px-2 py-1 rounded transition"
                  >
                    [Stage Work Order]
                  </button>
                </div>

                <div className="bg-[#06090e] border border-slate-800/80 rounded p-4 text-xs font-mono leading-relaxed h-[360px] overflow-y-auto space-y-3">
                  {loadingDiagnosis ? (
                    <div className="h-full flex flex-col items-center justify-center gap-3 text-slate-500">
                      <RefreshCw className="w-5 h-5 animate-spin text-cyan-500" />
                      <span>Synthesizing Telemetry & Vector Precedents...</span>
                    </div>
                  ) : diagnosis ? (
                    <div>
                      <div className="text-slate-300 whitespace-pre-wrap">{diagnosis.diagnostic_report}</div>
                      <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-400 space-y-1">
                        <div><strong className="text-slate-200">Allocated LOTO:</strong> {diagnosis.loto_protocol}</div>
                        <div><strong className="text-slate-200">Requisitioned Parts:</strong> {diagnosis.parts_allocated.join(', ') || 'None staged'}</div>
                        <div>
                          <strong className="text-slate-200">Precedent:</strong>{' '}
                          {diagnosis.matched_cmms_record
                            ? `${diagnosis.matched_cmms_record} (cosine ${diagnosis.similarity_score.toFixed(3)})`
                            : 'No record above similarity threshold'}
                        </div>
                        <div>
                          <strong className="text-slate-200">Dispatch path:</strong> {diagnosis.path}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center text-slate-600">
                      Click a triage chip or submit a query to execute diagnostic pipeline.
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-4 flex gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter industrial inquiry or maintenance command..."
                  className="flex-1 bg-slate-950 border border-slate-800 rounded px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
                  onKeyDown={(e) => e.key === 'Enter' && handleDiagnose()}
                />
                <button
                  onClick={() => handleDiagnose()}
                  disabled={loadingDiagnosis}
                  className="bg-cyan-600 hover:bg-cyan-500 text-black px-4 py-2 rounded text-xs font-mono font-bold flex items-center gap-1.5 transition disabled:opacity-50"
                >
                  <Send className="w-3.5 h-3.5" />
                  Run
                </button>
              </div>
            </div>

            {diagnosis && (
              <div className="bg-[#090d16] border border-slate-800 rounded overflow-hidden">
                <button
                  onClick={() => setTraceOpen(!traceOpen)}
                  className="w-full px-4 py-2.5 bg-slate-900/50 flex justify-between items-center text-xs font-mono text-slate-300 hover:bg-slate-900 transition"
                >
                  <span className="flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-amber-400" />
                    Tool Execution Pipeline & Kernel Trace
                  </span>
                  {traceOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {traceOpen && (
                  <div className="p-4 space-y-2 text-xs font-mono bg-black/40 border-t border-slate-800">
                    <div className="flex justify-between text-slate-400">
                      <span>Kinematic DSP (SK band, envelope, kurtosis):</span>
                      <span className="text-emerald-400">{diagnosis.execution_trace.cusignal_dsp_ms} ms</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>TF-IDF Vector Search (ISO 14224 CMMS):</span>
                      <span className="text-emerald-400">{diagnosis.execution_trace.cuvs_rag_retrieval_ms} ms</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Report Generation ({diagnosis.path}):</span>
                      <span className="text-emerald-400">{diagnosis.execution_trace.llm_ttft_ms} ms</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Output Verification Checks:</span>
                      <span className={checksPassed === checksTotal ? 'text-emerald-400' : 'text-rose-400'}>
                        {checksPassed}/{checksTotal} passed
                      </span>
                    </div>
                    <div className="pt-2 border-t border-slate-800 flex justify-between font-bold">
                      <span className="text-slate-200">Total Loop Latency:</span>
                      <span className="text-cyan-400">{diagnosis.execution_trace.total_pipeline_ms} ms</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* 4. BOTTOM PANEL: BENCHMARK MATRIX & WORK ORDER EXPORT */}
        <div className="bg-[#090d16] border border-slate-800 rounded p-5 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="w-full md:w-3/4">
            <h4 className="text-xs font-mono uppercase tracking-wider font-semibold text-slate-300 mb-3 flex items-center gap-2">
              <Server className="w-4 h-4 text-cyan-400" />
              Compute Profile (Measured CPU vs Projected NVIDIA Stack)
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-500">
                    <th className="pb-2">Workload Pipeline Stage</th>
                    <th className="pb-2">Measured CPU (x86_64)</th>
                    <th className="pb-2">Projected NVIDIA Stack</th>
                    <th className="pb-2 text-right">Projected Gain</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  <tr>
                    <td className="py-2">12 kHz CWRU DSP (envelope &amp; 4th-moment kurtosis)</td>
                    <td className="py-2 text-emerald-400">5.92 ms</td>
                    <td className="py-2 text-slate-500">1.18 ms (cuSignal)</td>
                    <td className="py-2 text-right font-bold text-slate-500">5.0x</td>
                  </tr>
                  <tr>
                    <td className="py-2">ISO-14224 Unstructured Vector Retrieval</td>
                    <td className="py-2 text-emerald-400">0.62 ms</td>
                    <td className="py-2 text-slate-500">0.82 ms (cuVS)</td>
                    <td className="py-2 text-right font-bold text-slate-500">n/a</td>
                  </tr>
                  <tr>
                    <td className="py-2">Generative Maintenance Plan Inference</td>
                    <td className="py-2 text-slate-500">not benchmarked</td>
                    <td className="py-2 text-slate-500">15.5 ms (TensorRT-LLM FP8)</td>
                    <td className="py-2 text-right font-bold text-slate-500">n/a</td>
                  </tr>
                </tbody>
              </table>
              <p className="mt-2 text-[10px] font-mono text-slate-600">
                CPU column measured on this build (2 vCPU Xeon 2.8 GHz, 300 repetitions). NVIDIA column is a design
                projection — no GPU stage has been benchmarked. See paper, Tables VII and XIV.
              </p>
            </div>
          </div>

          <div className="w-full md:w-1/4 flex flex-col items-center justify-center p-4 bg-slate-950/60 border border-slate-800/80 rounded">
            <span className="text-[11px] font-mono text-slate-400 mb-2">Validated CMMS Dispatch Artifact</span>
            <button
              onClick={handleDownloadWorkOrder}
              className="w-full py-2.5 px-4 bg-emerald-600 hover:bg-emerald-500 text-black font-mono font-bold text-xs rounded flex items-center justify-center gap-2 transition"
            >
              <Download className="w-4 h-4" />
              Export ISO-14224 Work Order
            </button>
            <span className="text-[10px] font-mono text-slate-600 mt-2">Format: SAP PM BAPI JSON</span>
          </div>
        </div>
      </main>
    </div>
  );
}
