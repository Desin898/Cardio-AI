import React from 'react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#6988A6] via-[#7F9EB8] to-[#9CB5CB] text-white font-sans relative overflow-x-hidden selection:bg-[#C6F022] selection:text-black">
      {/* Background Subtle Grid & Radial Glow Accents */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-20"
        style={{
          backgroundImage: `radial-gradient(rgba(255, 255, 255, 0.4) 1px, transparent 1px)`,
          backgroundSize: '36px 36px',
        }}
      />
      <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[900px] h-[500px] bg-white/15 blur-[120px] rounded-full pointer-events-none" />

      {/* Header Navigation */}
      <header className="relative z-20 max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
        {/* Logo */}
        <a href="/" className="flex items-center space-x-3 group">
          <div className="w-9 h-9 rounded-full bg-[#C6F022] flex items-center justify-center shadow-[0_0_20px_rgba(198,240,34,0.6)] group-hover:scale-105 transition-transform">
            <span className="w-3.5 h-3.5 rounded-full bg-[#0F291E] animate-pulse" />
          </div>
          <span className="text-2xl font-bold tracking-tight text-white drop-shadow-sm">
            Cardio<span className="text-[#C6F022]">-AI</span>
          </span>
        </a>

        {/* Nav Links */}
        <nav className="hidden md:flex items-center space-x-8 text-sm font-medium text-white/80">
          <a href="#about" className="hover:text-white transition-colors">About</a>
          <a href="/pre_screening.html" className="hover:text-[#C6F022] transition-colors">CKM Screening</a>
          <a href="/upload_ecg.html" className="hover:text-[#C6F022] transition-colors">12-Lead ECG</a>
          <a href="/angiogram_processing.html" className="hover:text-[#C6F022] transition-colors">Angiogram QCA</a>
          <a href="/docs" className="hover:text-white transition-colors">API & Support</a>
        </nav>

        {/* Action Buttons */}
        <div className="flex items-center space-x-4">
          <a
            href="/patient_login.html"
            className="px-5 py-2.5 rounded-full text-xs font-semibold tracking-wide uppercase bg-white/10 hover:bg-white/20 border border-white/25 backdrop-blur-md transition-all shadow-sm hover:shadow"
          >
            Patient Portal
          </a>
          <a
            href="/doctor_login.html"
            className="px-5 py-2.5 rounded-full text-xs font-semibold tracking-wide uppercase bg-white text-[#1B354C] hover:bg-[#C6F022] hover:text-[#0F291E] transition-all shadow-lg hover:shadow-[0_0_25px_rgba(198,240,34,0.5)]"
          >
            Doctor Portal
          </a>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-6 pb-20">
        {/* Upper Hero Text & Labels */}
        <div className="relative text-center">
          {/* Subheader Badges */}
          <div className="flex items-center justify-between text-xs font-semibold tracking-widest text-white/70 uppercase max-w-5xl mx-auto mb-4">
            <span className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-[#C6F022]" />
              <span>HEALTHCARE INNOVATOR</span>
            </span>
            <span>AI CORONARY ENGINE · SINCE 2026</span>
          </div>

          {/* Massive HEART SYNC Background Title */}
          <h1 className="text-7xl md:text-9xl font-extrabold tracking-widest text-white/90 uppercase drop-shadow-lg select-none">
            HEART <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-white/80 to-white/40">SYNC</span>
          </h1>

          {/* Centerpiece 3D Glass Heart Render */}
          <div className="relative -mt-20 md:-mt-32 flex justify-center items-center">
            <div className="relative w-[340px] md:w-[440px] h-[340px] md:h-[440px] flex justify-center items-center">
              {/* Glass Heart Image */}
              <img
                src="/static/images/glass_heart.jpg"
                alt="3D Translucent Glass Heart"
                className="w-full h-full object-contain filter drop-shadow-[0_20px_50px_rgba(0,30,60,0.4)] hover:scale-105 transition-transform duration-700 ease-out"
              />
              {/* Floating Glowing CTA Pill Overlay */}
              <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
                <a
                  href="/pre_screening.html"
                  className="px-8 py-3.5 rounded-full bg-[#C6F022] hover:bg-[#b5e016] text-[#0F291E] font-bold text-sm tracking-wide shadow-[0_0_30px_rgba(198,240,34,0.7)] hover:shadow-[0_0_40px_rgba(198,240,34,0.9)] transition-all flex items-center space-x-2 group whitespace-nowrap"
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
              <h3 className="text-xl font-bold uppercase tracking-wider text-white mb-2 leading-snug">
                GLOBAL EXPERTISE<br />YOU CAN TRUST
              </h3>
              <p className="text-xs text-white/70 leading-relaxed">
                Multimodal AI integration for CKM Biomarker Triage, 12-Lead ResNet34 ECG, and Quantitative Coronary Angiography (QCA).
              </p>
            </div>
          </div>

          {/* Subtitle */}
          <p className="mt-6 text-sm md:text-base text-white/80 font-medium max-w-2xl mx-auto leading-relaxed">
            Multimodal AI-Driven Coronary Disease Detection & Decision Support System
          </p>
        </div>

        {/* Bottom Info Cards (3-Column Grid) */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
          
          {/* Left Card (Dark Navy Glass) */}
          <div className="bg-[#122A3F]/80 backdrop-blur-xl border border-white/15 rounded-3xl p-6 shadow-2xl flex flex-col justify-between hover:border-white/30 transition-all duration-300 group">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-bold tracking-widest text-[#C6F022] uppercase">
                  CKM STAGE 0-4 TRIAGE
                </span>
                <span className="px-3 py-1 rounded-full bg-white/10 text-[10px] text-white/80 font-semibold border border-white/10">
                  ★ Top-Rated AI Model
                </span>
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Patient Triage Journey</h3>
              <p className="text-xs text-white/70 leading-relaxed mb-6">
                Rapid non-invasive prescreening evaluating HbA1c, Troponin I, Blood Pressure, and Lipid panels with SHAP feature contribution.
              </p>

              {/* Smartphone Mockup */}
              <div className="bg-gradient-to-b from-[#1C3B56] to-[#0D1F30] rounded-2xl p-4 border border-white/10 shadow-inner flex flex-col items-center text-center relative overflow-hidden">
                <div className="w-16 h-1 bg-white/20 rounded-full mb-3" />
                <span className="text-[10px] font-semibold text-white/60 tracking-wider uppercase mb-1">
                  YOUR HEALTH JOURNEY
                </span>
                <div className="w-14 h-14 rounded-full bg-[#C6F022]/20 border border-[#C6F022]/40 flex items-center justify-center my-2 group-hover:scale-110 transition-transform">
                  <span className="text-xl">🫀</span>
                </div>
                <a
                  href="/pre_screening.html"
                  className="mt-2 px-4 py-1.5 rounded-full bg-[#C6F022] text-[#0F291E] text-[11px] font-bold shadow-md hover:bg-white transition-colors"
                >
                  Start Now
                </a>
              </div>
            </div>
            
            <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between text-xs text-white/60">
              <span>Cardio-Kidney-Metabolic Protocol</span>
              <a href="/pre_screening.html" className="text-[#C6F022] hover:underline font-semibold">Explore →</a>
            </div>
          </div>

          {/* Center Card (Clean White Glass) */}
          <div className="bg-white/95 backdrop-blur-xl border border-white rounded-3xl p-6 shadow-2xl text-[#162A3B] flex flex-col justify-between hover:shadow-[0_25px_60px_rgba(0,0,0,0.25)] transition-all duration-300">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-bold tracking-widest text-[#2A7CFF] uppercase">
                  12-LEAD RESNET34 ECG
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-800 text-[10px] font-bold">
                  GroupKFold Validated
                </span>
              </div>

              <div className="flex items-baseline space-x-2 mb-2">
                <span className="text-5xl font-extrabold text-[#162A3B] tracking-tight">98.8%</span>
                <span className="text-xs font-semibold text-slate-500">Diagnostic Accuracy</span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed mb-6">
                MultiBranch 1D-ResNet34 backbone detecting LAD, LCx, RCA, and LMCA multi-vessel ischemia with softmax probability distribution.
              </p>

              {/* Doctor Avatar Card */}
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

            <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span>LMCA & STEMI Triage Protocol</span>
              <a href="/upload_ecg.html" className="text-blue-600 hover:underline font-bold">Upload ECG →</a>
            </div>
          </div>

          {/* Right Card (Soft Slate Glass) */}
          <div className="bg-[#597B9A]/80 backdrop-blur-xl border border-white/20 rounded-3xl p-6 shadow-2xl flex flex-col justify-between hover:border-white/40 transition-all duration-300 group">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-bold tracking-widest text-[#C6F022] uppercase">
                  DEEPSA & QCA ENGINE
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-white/10 text-white text-[10px] font-semibold border border-white/15">
                  Automated Pipeline
                </span>
              </div>

              <h3 className="text-xl font-bold text-white mb-2">Automated QCA Profiling</h3>
              <p className="text-xs text-white/80 leading-relaxed mb-6">
                Medial Axis Transform vessel centerline skeletonization, perpendicular diameter profiling, stenosis percentage calculation, and bottleneck visualization.
              </p>

              {/* 3D Glass Orbital Sphere */}
              <div className="relative h-32 rounded-2xl overflow-hidden flex items-center justify-center bg-black/20 border border-white/10">
                <img
                  src="/static/images/qca_glass_globe.jpg"
                  alt="3D QCA Orbital Sphere"
                  className="w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-3">
                  <span className="text-[11px] font-bold text-white tracking-wide">
                    Stenosis Bottleneck Detection & Grading
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between text-xs text-white/70">
              <span>DeepSA Segmentation</span>
              <a href="/angiogram_processing.html" className="text-[#C6F022] hover:underline font-semibold">QCA Engine →</a>
            </div>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/15 py-8 text-center text-xs text-white/60">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p>© 2026 Cardio-AI · Multimodal AI-Driven Coronary Disease Detection System</p>
          <div className="flex items-center space-x-6">
            <a href="/pre_screening.html" className="hover:text-white transition-colors">Prescreening</a>
            <a href="/upload_ecg.html" className="hover:text-white transition-colors">ECG Risk</a>
            <a href="/angiogram_processing.html" className="hover:text-white transition-colors">Angiogram QCA</a>
            <a href="/docs" className="hover:text-white transition-colors">FastAPI Docs</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
