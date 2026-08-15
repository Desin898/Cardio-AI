import React from 'react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0B1320] via-[#101C2E] to-[#0A101D] text-white font-sans relative overflow-x-hidden selection:bg-[#C6F022] selection:text-black">
      {/* Background Grid & Radial Glow Overlay */}
      <div 
        className="fixed inset-0 pointer-events-none opacity-20 z-0"
        style={{
          backgroundImage: `radial-gradient(rgba(255, 255, 255, 0.3) 1px, transparent 1px)`,
          backgroundSize: '36px 36px',
        }}
      />
      <div className="fixed -top-40 left-1/2 -translate-x-1/2 w-[900px] h-[550px] bg-[#00D2FF]/10 blur-[150px] rounded-full pointer-events-none z-0" />
      <div className="fixed bottom-10 right-10 w-[500px] h-[500px] bg-[#00E676]/10 blur-[160px] rounded-full pointer-events-none z-0" />

      {/* HEADER NAVIGATION */}
      <header className="relative z-20 max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
        {/* Logo */}
        <a href="/" className="flex items-center space-x-3 group">
          <div className="w-10 h-10 rounded-full bg-[#C6F022] flex items-center justify-center shadow-[0_0_20px_rgba(198,240,34,0.6)] group-hover:scale-105 transition-transform">
            <span className="w-3.5 h-3.5 rounded-full bg-[#0B1320] animate-pulse" />
          </div>
          <span className="text-2xl font-extrabold tracking-tight text-white drop-shadow-sm">
            Cardio<span className="text-[#C6F022]">-AI</span>
          </span>
        </a>

        {/* Nav Links */}
        <nav className="hidden md:flex items-center space-x-8 text-sm font-semibold text-slate-300">
          <a href="#about" className="hover:text-white transition-colors">About</a>
          <a href="/prescreening" className="hover:text-[#C6F022] transition-colors">CKM Screening</a>
          <a href="/upload_ecg.html" className="hover:text-[#C6F022] transition-colors">12-Lead ECG</a>
          <a href="/angiogram_qca" className="hover:text-[#C6F022] transition-colors">Angiogram QCA</a>
          <a href="/docs" className="hover:text-white transition-colors">API Docs</a>
        </nav>

        {/* Dual Portal Action Buttons */}
        <div className="flex items-center space-x-3">
          <a
            href="/prescreening"
            className="px-5 py-2.5 rounded-full text-xs font-semibold tracking-wide uppercase bg-white/10 hover:bg-white/20 border border-white/25 backdrop-blur-md transition-all shadow-sm"
          >
            Patient Portal
          </a>
          <a
            href="/doctor_dashboard"
            className="px-5 py-2.5 rounded-full text-xs font-bold tracking-wide uppercase bg-gradient-to-r from-[#00E676] to-[#00D2FF] text-black hover:scale-105 transition-all shadow-lg hover:shadow-[0_0_25px_rgba(0,230,118,0.5)]"
          >
            Doctor Portal
          </a>
        </div>
      </header>

      {/* HERO SECTION */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-4 pb-20">
        <div className="relative text-center">
          {/* Subheader Badges */}
          <div className="flex items-center justify-between text-xs font-bold tracking-widest text-slate-400 uppercase max-w-5xl mx-auto mb-3">
            <span className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#C6F022]" />
              <span className="text-slate-300">HEALTHCARE INNOVATOR</span>
            </span>
            <span className="text-slate-400">AI CORONARY ENGINE · SINCE 2026</span>
          </div>

          {/* Massive HEART SYNC Metallic Title */}
          <h1 className="relative z-0 text-7xl md:text-9xl font-extrabold tracking-widest uppercase select-none drop-shadow-2xl">
            HEART <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-200 to-slate-400">SYNC</span>
          </h1>

          {/* Centerpiece 3D Glass Heart & CTA Button */}
          <div className="relative z-10 -mt-20 md:-mt-32 flex justify-center items-center">
            <div className="relative w-[340px] md:w-[460px] h-[340px] md:h-[460px] flex justify-center items-center">
              {/* Glass Heart Image with Seamless Radial Blend Mask */}
              <img
                src="/static/images/glass_heart.jpg"
                alt="3D Translucent Glass Heart"
                className="w-full h-full object-contain relative z-10 filter drop-shadow-[0_25px_60px_rgba(0,210,255,0.35)] hover:scale-105 transition-transform duration-700 ease-out"
                style={{
                  maskImage: 'radial-gradient(circle at center, black 50%, transparent 88%)',
                  WebkitMaskImage: 'radial-gradient(circle at center, black 50%, transparent 88%)',
                  mixBlendMode: 'lighten',
                }}
              />
              {/* Floating Glowing Lime CTA Pill Overlay */}
              <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20">
                <a
                  href="/prescreening"
                  className="px-8 py-3.5 rounded-full bg-[#C6F022] hover:bg-[#b5e016] text-[#0B1320] font-extrabold text-sm tracking-wide shadow-[0_0_30px_rgba(198,240,34,0.7)] transition-all flex items-center space-x-2.5 group whitespace-nowrap"
                >
                  <span>Start Cardiac Screening</span>
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </a>
              </div>
            </div>

            {/* Right Side Callout */}
            <div className="hidden lg:block absolute right-4 top-1/2 -translate-y-1/2 text-right max-w-xs">
              <h3 className="text-xl font-black uppercase tracking-wider text-white mb-2 leading-snug">
                GLOBAL EXPERTISE<br />YOU CAN TRUST
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Multimodal AI integration for CKM Biomarker Triage, 12-Lead ResNet34 ECG, and Quantitative Coronary Angiography (QCA).
              </p>
            </div>
          </div>

          {/* Subtitle */}
          <p className="mt-6 text-sm md:text-base text-slate-300 font-medium max-w-2xl mx-auto leading-relaxed">
            Multimodal AI-Driven Coronary Disease Detection & Decision Support System
          </p>
        </div>

        {/* 3-COLUMN BOTTOM GLASS CARDS SHOWCASE */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
          {/* Card 1 (Dark Navy Glass) */}
          <div className="bg-[#0D1B2A]/85 backdrop-blur-xl border border-white/12 rounded-3xl p-6 shadow-2xl flex flex-col justify-between hover:border-white/35 transition-all duration-300 group">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-extrabold tracking-widest text-[#C6F022] uppercase">
                  CKM STAGE 0-4 TRIAGE
                </span>
                <span className="px-3 py-1 rounded-full bg-white/10 text-[10px] text-white/90 font-semibold border border-white/10">
                  ★ Top-Rated AI Model
                </span>
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Patient Triage Journey</h3>
              <p className="text-xs text-slate-300 leading-relaxed mb-6">
                Rapid non-invasive prescreening evaluating HbA1c, Troponin I, Blood Pressure, and Lipid panels with SHAP feature contribution.
              </p>

              {/* Smartphone Mockup */}
              <div className="bg-gradient-to-b from-[#162B42] to-[#0A1624] rounded-2xl p-4 border border-white/10 shadow-inner flex flex-col items-center text-center relative overflow-hidden">
                <div className="w-16 h-1 bg-white/20 rounded-full mb-3" />
                <span className="text-[10px] font-semibold text-slate-400 tracking-wider uppercase mb-1">
                  YOUR HEALTH JOURNEY
                </span>
                <div className="w-14 h-14 rounded-full bg-[#C6F022]/20 border border-[#C6F022]/40 flex items-center justify-center my-2 group-hover:scale-110 transition-transform">
                  <span className="text-2xl">🫀</span>
                </div>
                <a
                  href="/prescreening"
                  className="mt-2 px-5 py-1.5 rounded-full bg-[#C6F022] text-[#0B1320] text-[11px] font-extrabold shadow-md hover:bg-white transition-colors"
                >
                  Start Now
                </a>
              </div>
            </div>
            
            <div className="mt-5 pt-4 border-t border-white/10 flex items-center justify-between text-xs text-slate-400">
              <span>Cardio-Kidney-Metabolic Protocol</span>
              <a href="/prescreening" className="text-[#C6F022] hover:underline font-semibold">Explore →</a>
            </div>
          </div>

          {/* Card 2 (Crisp White Glass) */}
          <div className="bg-white/95 backdrop-blur-xl border border-white rounded-3xl p-6 shadow-2xl text-[#101C2E] flex flex-col justify-between hover:shadow-[0_25px_60px_rgba(0,0,0,0.3)] transition-all duration-300">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-extrabold tracking-widest text-[#2A7CFF] uppercase">
                  12-LEAD RESNET34 ECG
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-800 text-[10px] font-bold">
                  GroupKFold Validated
                </span>
              </div>

              <div className="flex items-baseline space-x-2 mb-2">
                <span className="text-5xl font-extrabold text-[#101C2E] tracking-tight">98.8%</span>
                <span className="text-xs font-semibold text-slate-500">Diagnostic Accuracy</span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed mb-6">
                MultiBranch 1D-ResNet34 backbone detecting LAD, LCx, RCA, and LMCA multi-vessel ischemia with softmax probability distribution.
              </p>

              {/* Doctor Avatar Info Box */}
              <div className="flex items-center space-x-4 bg-slate-50 p-3 rounded-2xl border border-slate-100">
                <img
                  src="/static/images/clinical_doctor_avatar.jpg"
                  alt="Doctor Specialist"
                  className="w-14 h-14 rounded-full object-cover border-2 border-white shadow-sm"
                />
                <div>
                  <h4 className="text-sm font-bold text-slate-800">Dr. Sarah Jenkins, M.D.</h4>
                  <p className="text-[11px] font-semibold text-slate-500">10+ YEARS CLINICAL EXPERTISE</p>
                  <p className="text-[10px] text-blue-600 font-medium mt-0.5">Cardiology Decision Support</p>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span>LMCA & STEMI Triage Protocol</span>
              <a href="/upload_ecg.html" className="text-blue-600 hover:underline font-bold">Upload ECG →</a>
            </div>
          </div>

          {/* Card 3 (Soft Slate Glass) */}
          <div className="bg-[#1E2E42]/85 backdrop-blur-xl border border-white/15 rounded-3xl p-6 shadow-2xl flex flex-col justify-between hover:border-white/40 transition-all duration-300 group">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-extrabold tracking-widest text-[#C6F022] uppercase">
                  DEEPSA & QCA ENGINE
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-white/10 text-white text-[10px] font-semibold border border-white/15">
                  Automated Pipeline
                </span>
              </div>

              <h3 className="text-xl font-bold text-white mb-2">Automated QCA Profiling</h3>
              <p className="text-xs text-slate-300 leading-relaxed mb-6">
                Medial Axis Transform vessel centerline skeletonization, perpendicular diameter profiling, stenosis percentage calculation, and bottleneck visualization.
              </p>

              {/* 3D Orbital Sphere Box */}
              <div className="relative h-32 rounded-2xl overflow-hidden flex items-center justify-center bg-black/30 border border-white/10">
                <img
                  src="/static/images/qca_glass_globe.jpg"
                  alt="3D QCA Orbital Sphere"
                  className="w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent flex items-end p-3">
                  <span className="text-[11px] font-bold text-white tracking-wide">
                    Stenosis Bottleneck Detection & Grading
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-4 border-t border-white/10 flex items-center justify-between text-xs text-slate-400">
              <span>DeepSA Segmentation</span>
              <a href="/angiogram_qca" className="text-[#C6F022] hover:underline font-semibold">QCA Engine →</a>
            </div>
          </div>
        </div>
      </main>

      {/* FOOTER */}
      <footer className="relative z-10 border-t border-white/15 py-8 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p>© 2026 Cardio-AI · Multimodal AI-Driven Coronary Disease Detection System</p>
          <div className="flex items-center space-x-6">
            <a href="/prescreening" className="hover:text-white transition-colors">Prescreening</a>
            <a href="/doctor_dashboard" className="hover:text-[#00E676] transition-colors font-semibold">Doctor Portal</a>
            <a href="/upload_ecg.html" className="hover:text-white transition-colors">ECG Risk</a>
            <a href="/angiogram_qca" className="hover:text-white transition-colors">Angiogram QCA</a>
            <a href="/docs" className="hover:text-white transition-colors">API Docs</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
