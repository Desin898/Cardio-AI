import React, { useEffect, useRef } from 'react';

export default function DoctorDashboard() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    let x = 0;
    const points = [];
    const height = canvas.height;
    const width = canvas.width;
    let animationFrameId;

    const getEcgY = (t) => {
      const phase = t % 100;
      if (phase > 35 && phase < 40) return height / 2 - 6;
      if (phase >= 40 && phase < 43) return height / 2 + 3;
      if (phase >= 43 && phase < 47) return height / 2 - 24;
      if (phase >= 47 && phase < 50) return height / 2 + 8;
      if (phase >= 55 && phase < 65) return height / 2 - 8;
      return height / 2;
    };

    const draw = () => {
      ctx.fillStyle = '#030508';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = 'rgba(0, 230, 118, 0.08)';
      ctx.lineWidth = 1;
      for (let gx = 0; gx < width; gx += 16) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
        ctx.stroke();
      }
      for (let gy = 0; gy < height; gy += 12) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
        ctx.stroke();
      }

      ctx.strokeStyle = '#00E676';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#00E676';
      ctx.shadowBlur = 8;
      ctx.beginPath();

      for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      const y = getEcgY(x);
      points.push({ x: x % width, y });
      if (points.length > width) {
        points.shift();
      }

      x += 2;
      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#06080D] text-slate-200 font-sans selection:bg-[#00E676] selection:text-black flex flex-col md:flex-row overflow-x-hidden">
      <div 
        className="fixed inset-0 pointer-events-none z-0 opacity-15"
        style={{
          backgroundImage: `radial-gradient(rgba(0, 210, 255, 0.3) 1px, transparent 1px)`,
          backgroundSize: '32px 32px',
        }}
      />
      <div className="fixed top-0 left-1/4 w-[600px] h-[600px] bg-[#00D2FF]/5 blur-[160px] rounded-full pointer-events-none z-0" />
      <div className="fixed bottom-0 right-1/4 w-[600px] h-[600px] bg-[#00E676]/5 blur-[160px] rounded-full pointer-events-none z-0" />

      {/* LEFT SIDEBAR NAVIGATION */}
      <aside className="relative z-30 w-full md:w-64 bg-[#0A0D14] border-b md:border-b-0 md:border-r border-white/10 flex-shrink-0 flex flex-col justify-between p-4">
        <div>
          <a href="/" className="flex items-center space-x-3 px-2 py-3 mb-6 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00E676] to-[#00D2FF] flex items-center justify-center shadow-[0_0_20px_rgba(0,230,118,0.4)] group-hover:scale-105 transition-transform">
              <svg className="w-5 h-5 text-black" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
              </svg>
            </div>
            <div>
              <div className="text-lg font-black tracking-wider text-white uppercase flex items-center gap-1.5">
                CARDIO<span className="text-[#00E676]">-AI</span>
              </div>
              <div className="text-[10px] font-mono text-slate-400 tracking-widest uppercase">DOCTOR COMMAND CENTER</div>
            </div>
          </a>

          <nav className="space-y-1.5 font-medium text-xs">
            <a href="/doctor_dashboard" className="flex items-center space-x-3 px-3.5 py-3 rounded-xl bg-[#00E676]/10 text-[#00E676] border-l-4 border-[#00E676] font-semibold transition-all">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <span>Cath Lab Dashboard</span>
            </a>
            <a href="/prescreening" className="flex items-center space-x-3 px-3.5 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-all group">
              <svg className="w-4 h-4 group-hover:text-[#00D2FF]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>CKM Patient Triage</span>
            </a>
            <a href="/upload_ecg.html" className="flex items-center space-x-3 px-3.5 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-all group">
              <svg className="w-4 h-4 group-hover:text-[#00E676]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span>12-Lead ECG Analysis</span>
            </a>
            <a href="/angiogram_qca" className="flex items-center space-x-3 px-3.5 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-all group">
              <svg className="w-4 h-4 group-hover:text-[#00D2FF]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              <span>Angiogram QCA Workspace</span>
            </a>
            <a href="/patient_dashboard.html" className="flex items-center space-x-3 px-3.5 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-all group">
              <svg className="w-4 h-4 group-hover:text-[#FF3D57]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>Diagnostic Reports</span>
            </a>
            <a href="/" className="flex items-center space-x-3 px-3.5 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-all group">
              <svg className="w-4 h-4 group-hover:text-slate-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              <span>Public Landing Page</span>
            </a>
          </nav>
        </div>

        <div className="mt-8 pt-4 border-t border-white/10 text-[11px] font-mono text-slate-400 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-[#00E676] animate-ping" />
              <span className="text-slate-300">FastAPI Core</span>
            </span>
            <span className="text-[#00E676]">v2.0 ACTIVE</span>
          </div>
          <div className="flex items-center justify-between text-[10px]">
            <span>Cath-Lab Net</span>
            <span className="text-[#00D2FF]">CONNECTED</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTAINER */}
      <div className="relative z-10 flex-1 flex flex-col min-w-0 overflow-y-auto">
        <header className="bg-[#0A0D14]/90 backdrop-blur-md border-b border-white/10 px-6 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 sticky top-0 z-40">
          <div className="relative w-full md:w-96">
            <svg className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input 
              type="text" 
              placeholder="Search Patient ID (e.g. PT-89421-B), MRN, or Chief Complaint..." 
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-[#06080D] border border-white/10 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-[#00D2FF] focus:ring-1 focus:ring-[#00D2FF] transition-all"
            />
          </div>

          <div className="flex items-center space-x-5 text-xs">
            <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-[#141A24] border border-white/10 font-mono">
              <span className="text-slate-300">AUG 15, 2026</span>
              <span className="text-slate-600">|</span>
              <span className="text-[#00E676]">SHIFT: CATH-1 ALPHA</span>
            </div>

            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-[#00E676]/10 border border-[#00E676]/30 text-[#00E676] font-mono">
              <span className="w-2 h-2 rounded-full bg-[#00E676] animate-pulse" />
              <span className="font-semibold">SESSION: #8C6E-7739</span>
            </div>

            <div className="flex items-center space-x-3 pl-2 border-l border-white/10">
              <img 
                src="/static/images/clinical_doctor_avatar.jpg" 
                alt="Attending Doctor" 
                className="w-8 h-8 rounded-full object-cover border-2 border-[#00E676]"
              />
              <div className="hidden xl:block">
                <div className="font-bold text-white leading-tight">Dr. Sarah Jenkins, M.D.</div>
                <div className="text-[10px] text-slate-400">Chief Interventionalist</div>
              </div>
            </div>
          </div>
        </header>

        {/* 3-COLUMN COMMAND GRID */}
        <main className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* LEFT COLUMN (3 cols) */}
          <section className="lg:col-span-3 space-y-6">
            <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-5 border border-white/10 relative overflow-hidden">
              <div className="flex items-center justify-between mb-3">
                <span className="px-2.5 py-1 rounded-md bg-[#00D2FF]/10 text-[#00D2FF] text-[10px] font-mono font-bold tracking-wider uppercase border border-[#00D2FF]/20">
                  CATH LAB 02 · ADMITTED
                </span>
                <span className="text-[11px] font-mono text-slate-400">#PT-89421-B</span>
              </div>

              <h2 className="text-xl font-extrabold text-white tracking-tight mb-1">Eleanor Vance</h2>
              <div className="text-xs text-slate-400 font-medium mb-4 flex items-center space-x-2">
                <span>62 Yrs</span>
                <span>•</span>
                <span>Female</span>
                <span>•</span>
                <span className="text-slate-300">MRN: 902-88-11</span>
              </div>

              <div className="bg-[#06080D] rounded-xl p-3 border border-white/10 space-y-1.5 text-xs">
                <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Admission Chief Complaint</div>
                <div className="font-medium text-slate-200 flex items-center space-x-2">
                  <span className="text-[#FF3D57]">⚠️</span>
                  <span>Syncope & Acute Substernal Chest Pain</span>
                </div>
                <div className="text-[10px] text-slate-500">Onset: 45 mins prior to triage</div>
              </div>
            </div>

            {/* 2x2 Metric Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-4 border border-white/10">
                <div className="flex items-center justify-between text-slate-400 text-[10px] font-mono mb-2">
                  <span>BLOOD PRESSURE</span>
                  <span className="text-[#00E676]">🫀</span>
                </div>
                <div className="text-lg font-mono font-extrabold text-white tracking-tight">116/70</div>
                <div className="text-[10px] text-[#00E676] font-medium mt-1">Normal Systolic</div>
              </div>

              <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-4 border border-white/10">
                <div className="flex items-center justify-between text-slate-400 text-[10px] font-mono mb-2">
                  <span>HEART RATE</span>
                  <span className="text-[#00E676] animate-pulse">❤️</span>
                </div>
                <div className="text-lg font-mono font-extrabold text-[#00E676] tracking-tight flex items-baseline gap-1">
                  <span>64</span>
                  <span className="text-xs font-normal text-slate-400">BPM</span>
                </div>
                <div className="text-[10px] text-slate-400 font-medium mt-1">Normal Sinus</div>
              </div>

              <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-4 border border-white/10">
                <div className="flex items-center justify-between text-slate-400 text-[10px] font-mono mb-2">
                  <span>GLUCOSE / HbA1c</span>
                  <span className="text-[#00D2FF]">💧</span>
                </div>
                <div className="text-lg font-mono font-extrabold text-[#00D2FF] tracking-tight">6.4%</div>
                <div className="text-[10px] text-slate-400 font-medium mt-1">128 mg/dL</div>
              </div>

              <div className="bg-[#FF3D57]/10 backdrop-blur-xl rounded-2xl p-4 border-2 border-[#FF3D57]/40">
                <div className="flex items-center justify-between text-slate-400 text-[10px] font-mono mb-2">
                  <span>hs-TROPONIN I</span>
                  <span className="text-[#FF3D57] animate-bounce">🔔</span>
                </div>
                <div className="text-lg font-mono font-extrabold text-[#FF3D57] tracking-tight">0.04</div>
                <div className="text-[10px] text-[#FF3D57] font-semibold mt-1">ng/mL · ELEVATED</div>
              </div>
            </div>

            {/* CKM Triage Profile */}
            <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-5 border border-white/10 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold tracking-wider text-slate-300 uppercase">CKM TRIAGE PROFILE</span>
                <span className="px-2 py-0.5 rounded bg-[#FFB300]/10 text-[#FFB300] text-[10px] font-mono font-bold border border-[#FFB300]/20">STAGE 2/3</span>
              </div>

              <div>
                <div className="flex items-baseline justify-between text-xs mb-1.5">
                  <span className="text-slate-400">Ischemic Risk Score</span>
                  <span className="font-mono font-bold text-[#FF3D57]">88.4%</span>
                </div>
                <div className="w-full h-2.5 rounded-full bg-[#06080D] overflow-hidden p-0.5 border border-white/10">
                  <div className="h-full rounded-full bg-gradient-to-r from-[#00E676] via-[#FFB300] to-[#FF3D57]" style={{ width: '88.4%' }} />
                </div>
              </div>

              <div className="text-[11px] text-slate-400 leading-relaxed bg-[#06080D] p-3 rounded-xl border border-white/10">
                <span className="text-white font-semibold">Key Biomarkers: </span>
                Elevated hs-Troponin I, HbA1c 6.4%, and ECG V2-V4 ST-segment deviation.
              </div>
            </div>
          </section>

          {/* CENTER COLUMN (6 cols) */}
          <section className="lg:col-span-6 space-y-6">
            <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-3xl p-6 relative overflow-hidden flex flex-col items-center justify-center min-h-[480px] border border-white/10">
              
              <div className="w-full flex items-center justify-between z-20 mb-2">
                <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-[#00E676]/10 border border-[#00E676]/30 text-[#00E676] text-xs font-semibold">
                  <span className="text-sm">✓</span>
                  <span>Clinically Validated</span>
                </div>
                <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-[#00D2FF]/10 border border-[#00D2FF]/30 text-[#00D2FF] text-xs font-mono">
                  <span>🫀</span>
                  <span>Vascular Age: 34-36 yrs</span>
                </div>
              </div>

              <div className="relative w-full max-w-[380px] h-[320px] flex items-center justify-center my-4">
                <img 
                  src="/static/images/glass_heart.jpg" 
                  alt="3D Holographic Heart Wireframe" 
                  className="w-64 h-64 object-contain filter drop-shadow-[0_0_35px_rgba(0,210,255,0.35)] hover:scale-105 transition-transform duration-700 z-10"
                />

                <div className="absolute top-6 right-2 z-20 bg-[#06080D]/90 border border-[#FF3D57] text-white px-2.5 py-1 rounded-lg text-[10px] font-mono shadow-lg backdrop-blur-md flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#FF3D57] animate-ping" />
                  <span>LAD: 85% Stenosis (Distal)</span>
                </div>

                <div className="absolute bottom-10 right-0 z-20 bg-[#06080D]/90 border border-[#00E676] text-white px-2.5 py-1 rounded-lg text-[10px] font-mono shadow-lg backdrop-blur-md flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#00E676]" />
                  <span>LCx: Patent</span>
                </div>

                <div className="absolute bottom-8 left-0 z-20 bg-[#06080D]/90 border border-[#FFB300] text-white px-2.5 py-1 rounded-lg text-[10px] font-mono shadow-lg backdrop-blur-md flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#FFB300]" />
                  <span>RCA: Mild Plaque</span>
                </div>

                <div className="absolute top-8 left-2 z-20 bg-[#06080D]/90 border border-[#00D2FF] text-white px-2.5 py-1 rounded-lg text-[10px] font-mono shadow-lg backdrop-blur-md flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#00D2FF]" />
                  <span>LMCA: Clear</span>
                </div>
              </div>

              {/* Live Canvas ECG Waveform */}
              <div className="w-full bg-[#06080D] rounded-2xl p-3 border border-white/10 z-20">
                <div className="flex items-center justify-between text-[11px] font-mono mb-1 text-slate-400">
                  <span className="flex items-center space-x-2 text-[#00E676]">
                    <span className="w-2 h-2 rounded-full bg-[#00E676] animate-pulse" />
                    <span className="font-bold">LEAD II ECG WAVEFORM</span>
                  </span>
                  <span>64 BPM · Normal Sinus Rhythm</span>
                </div>
                <canvas ref={canvasRef} width={500} height={54} className="w-full h-14 rounded-lg bg-[#030508]" />
              </div>

              {/* Primary Action Button */}
              <div className="mt-6 z-20 w-full sm:w-auto">
                <a 
                  href="/upload_ecg.html" 
                  className="w-full sm:w-auto px-8 py-3.5 rounded-full bg-gradient-to-r from-[#00E676] to-[#00D2FF] hover:from-[#00c865] hover:to-[#00b0d8] text-black font-extrabold text-sm tracking-wide shadow-[0_0_30px_rgba(0,230,118,0.5)] transition-all flex items-center justify-center space-x-3 group"
                >
                  <span>Initiate 12-Lead ECG / QCA Analysis</span>
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </a>
              </div>
            </div>

            {/* Quick Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <a href="/prescreening" className="bg-[#0F141C]/85 backdrop-blur-xl p-4 rounded-2xl border border-white/10 hover:border-[#00E676]/40 transition-all group">
                <div className="text-[#00E676] text-xs font-mono font-bold uppercase mb-1">CKM BIOMARKER</div>
                <h4 className="text-sm font-bold text-white group-hover:text-[#00E676] transition-colors">Risk Prescreening</h4>
                <p className="text-[11px] text-slate-400 mt-1">Multi-factor ensemble & SHAP explanation.</p>
              </a>

              <a href="/upload_ecg.html" className="bg-[#0F141C]/85 backdrop-blur-xl p-4 rounded-2xl border border-white/10 hover:border-[#00D2FF]/40 transition-all group">
                <div className="text-[#00D2FF] text-xs font-mono font-bold uppercase mb-1">1D-RESNET34</div>
                <h4 className="text-sm font-bold text-white group-hover:text-[#00D2FF] transition-colors">ECG Signal Model</h4>
                <p className="text-[11px] text-slate-400 mt-1">98.8% Accuracy on multi-lead data.</p>
              </a>

              <a href="/angiogram_qca" className="bg-[#0F141C]/85 backdrop-blur-xl p-4 rounded-2xl border border-white/10 hover:border-[#FF3D57]/40 transition-all group">
                <div className="text-[#FF3D57] text-xs font-mono font-bold uppercase mb-1">DEEPSA ENGINE</div>
                <h4 className="text-sm font-bold text-white group-hover:text-[#FF3D57] transition-colors">Angiogram QCA</h4>
                <p className="text-[11px] text-slate-400 mt-1">Stenosis bottleneck detection & vessel profiling.</p>
              </a>
            </div>
          </section>

          {/* RIGHT COLUMN (3 cols) */}
          <section className="lg:col-span-3 space-y-6">
            <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-5 border border-white/10 space-y-3">
              <div className="text-xs font-mono font-bold tracking-wider text-slate-400 uppercase">ATTENDING CARDIOLOGIST</div>
              <div className="flex items-center space-x-3">
                <img 
                  src="/static/images/clinical_doctor_avatar.jpg" 
                  alt="Dr. Sarah Jenkins" 
                  className="w-14 h-14 rounded-full object-cover border-2 border-[#00E676] shadow-md"
                />
                <div>
                  <h3 className="text-sm font-extrabold text-white">Dr. Sarah Jenkins, M.D.</h3>
                  <div className="text-[11px] text-[#00E676] font-medium">FACC Interventional Lead</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">10+ Yrs Cath Lab Experience</div>
                </div>
              </div>
              <div className="bg-[#06080D] p-2.5 rounded-xl border border-white/10 text-[11px] font-mono text-slate-300 flex items-center justify-between">
                <span>SHIFT SCHEDULE</span>
                <span className="text-[#00D2FF]">07:00 - 19:00</span>
              </div>
            </div>

            <div className="bg-[#0F141C]/85 backdrop-blur-xl rounded-2xl p-5 border border-white/10 space-y-3">
              <div className="flex items-center justify-between text-xs font-mono font-bold text-slate-400 uppercase">
                <span>CONSULTING SPECIALISTS</span>
                <span className="text-[#00E676]">3 ONLINE</span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div className="flex items-center justify-between p-2 rounded-xl bg-[#06080D] border border-white/5">
                  <div>
                    <div className="font-bold text-slate-200">Dr. Marcus Vance</div>
                    <div className="text-[10px] text-slate-400">Cardiovascular Surgery</div>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-[#00E676]/10 text-[#00E676] text-[10px] font-mono font-bold">AVAILABLE</span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-[#06080D] border border-white/5">
                  <div>
                    <div className="font-bold text-slate-200">Dr. Elena Rostova</div>
                    <div className="text-[10px] text-slate-400">QCA AI Specialist</div>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-[#00D2FF]/10 text-[#00D2FF] text-[10px] font-mono font-bold">ACTIVE SESSION</span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-[#06080D] border border-white/5">
                  <div>
                    <div className="font-bold text-slate-200">Dr. Aris Thorne</div>
                    <div className="text-[10px] text-slate-400">Electrophysiology</div>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-[#FFB300]/10 text-[#FFB300] text-[10px] font-mono font-bold">IN SURGERY</span>
                </div>
              </div>
            </div>

            <div className="bg-[#FF3D57]/10 backdrop-blur-xl rounded-2xl p-5 border-2 border-[#FF3D57]/60 space-y-3 relative overflow-hidden">
              <div className="flex items-center space-x-2 text-[#FF3D57] font-mono font-bold text-xs">
                <span className="animate-bounce">⚠️</span>
                <span>EMERGENCY CATH LAB ACTIVATION</span>
              </div>

              <h4 className="text-sm font-extrabold text-white">STAT Cath Lab Code Red</h4>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                Instant notification protocol for STEMI triage, direct bypass to QCA pipeline, and emergency cath team assembly.
              </p>

              <a 
                href="/angiogram_qca" 
                className="w-full py-2.5 px-4 rounded-xl bg-[#FF3D57] hover:bg-[#e02d46] text-white font-extrabold text-xs tracking-wider uppercase shadow-[0_0_20px_rgba(255,61,87,0.5)] transition-all flex items-center justify-center space-x-2"
              >
                <span>⚡</span>
                <span>TRIGGER EMERGENCY QCA</span>
              </a>
            </div>

          </section>
        </main>

        <footer className="mt-auto border-t border-white/10 py-6 px-6 text-xs text-slate-500 font-mono flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            © 2026 Cardio-AI · Cath Lab Clinical Command Center (Doctor Portal)
          </div>
          <div className="flex items-center space-x-6">
            <a href="/prescreening" className="hover:text-slate-300 transition-colors">Patient Triage</a>
            <a href="/upload_ecg.html" class="hover:text-slate-300 transition-colors">12-Lead ECG</a>
            <a href="/angiogram_qca" class="hover:text-slate-300 transition-colors">Angiogram QCA</a>
            <a href="/docs" class="hover:text-slate-300 transition-colors">API Specs</a>
          </div>
        </footer>
      </div>
    </div>
  );
}
