import React, { useState, useEffect } from 'react';

export default function ECGAnalysisView() {
  const [patientId, setPatientId] = useState('');
  const [age, setAge] = useState(58);
  const [gender, setGender] = useState('Male');
  const [ecgType, setEcgType] = useState('12_lead');
  const [notes, setNotes] = useState('');
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  
  const [status, setStatus] = useState('idle'); // 'idle' | 'analyzing' | 'complete'
  const [analysisResult, setAnalysisResult] = useState(null);

  useEffect(() => {
    const raw = localStorage.getItem('screeningFormData') || localStorage.getItem('screeningData');
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed.patient_id) setPatientId(parsed.patient_id);
        if (parsed.age) setAge(parsed.age);
        if (parsed.gender) setGender(parsed.gender);
      } catch (e) {}
    }
  }, []);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      if (selected.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (ev) => setFilePreview(ev.target.result);
        reader.readAsDataURL(selected);
      } else {
        setFilePreview(null);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      alert('Please select or drop an ECG signal or image file.');
      return;
    }

    setStatus('analyzing');

    const formData = new FormData();
    formData.append('ecg_file', file);
    formData.append('patient_id', patientId);
    formData.append('age', age);
    formData.append('gender', gender);
    formData.append('ecg_type', ecgType);
    formData.append('notes', notes);

    try {
      const response = await fetch('/api/v1/ecg/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);
      const data = await response.json();
      setAnalysisResult(data);
      setStatus('complete');
    } catch (err) {
      alert('ECG Analysis failed: ' + err.message);
      setStatus('idle');
    }
  };

  const qcaUrl = patientId ? `/angiogram_qca.html?patient_id=${encodeURIComponent(patientId)}` : '/angiogram_qca.html';

  return (
    <div className="min-h-screen bg-[#06080D] text-slate-200 font-sans p-6">
      <header className="max-w-7xl mx-auto flex items-center justify-between border-b border-white/10 pb-4 mb-8">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#00E676] to-[#00D2FF] flex items-center justify-center font-bold text-black">
            <i className="fa-solid fa-heart-pulse"></i>
          </div>
          <span className="text-xl font-extrabold text-white tracking-wider uppercase">CARDIO<span className="text-[#00E676]">-AI</span></span>
        </div>
        <a href={qcaUrl} className="text-xs font-semibold text-[#00D2FF] hover:underline flex items-center gap-1">
          <span>Angiography QCA Suite</span> &rarr;
        </a>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* LEFT FORM PANEL */}
        <div className="lg:col-span-5 bg-[#0F141C] p-6 rounded-3xl border border-white/10 space-y-4">
          <h2 className="text-sm font-mono font-bold text-[#00D2FF] uppercase tracking-wider">
            Patient & 12-Lead Signal Ingestion
          </h2>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="block text-slate-400 mb-1">Patient ID</label>
              <input type="text" value={patientId} onChange={(e) => setPatientId(e.target.value)} placeholder="PAT-8829" className="w-full p-2.5 rounded-xl bg-[#06080D] border border-white/10 text-white font-mono" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1">Age</label>
              <input type="number" value={age} onChange={(e) => setAge(e.target.value)} className="w-full p-2.5 rounded-xl bg-[#06080D] border border-white/10 text-white font-mono" />
            </div>
          </div>

          <div className="text-xs">
            <label className="block text-slate-400 mb-1">ECG Input File (.jpg, .png, .csv)</label>
            <input type="file" accept=".jpg,.jpeg,.png,.csv" onChange={handleFileChange} className="w-full text-slate-300 text-xs" />
            {file && <p className="mt-1 text-[11px] text-[#00E676] font-mono">Selected: {file.name}</p>}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={status === 'analyzing'}
            className="w-full py-3 rounded-full bg-gradient-to-r from-[#00D2FF] to-[#00E676] text-black font-extrabold text-xs uppercase tracking-wider hover:scale-105 transition-all shadow-[0_0_20px_rgba(0,210,255,0.4)]"
          >
            {status === 'analyzing' ? 'Running MultiBranch 1D-ResNet34...' : 'Analyze 12-Lead ECG'}
          </button>
        </div>

        {/* RIGHT DIAGNOSTIC PANEL */}
        <div className="lg:col-span-7">
          {status === 'idle' && (
            <div className="bg-[#0F141C] p-12 rounded-3xl border border-white/10 text-center space-y-3 flex flex-col items-center justify-center min-h-[420px]">
              <i className="fa-solid fa-wave-square text-4xl text-[#00D2FF]/40 mb-2"></i>
              <h3 className="text-lg font-bold text-white">No ECG Signal Processed</h3>
              <p className="text-xs text-slate-400 max-w-sm">Upload a .jpg/.png tracing or .csv signal matrix on the left to run ResNet34 inference.</p>
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-400 text-[10px] font-mono">STATUS: AWAITING SIGNAL</span>
            </div>
          )}

          {status === 'analyzing' && (
            <div className="bg-[#0F141C] p-12 rounded-3xl border border-[#00D2FF]/30 text-center space-y-3 flex flex-col items-center justify-center min-h-[420px]">
              <i className="fa-solid fa-circle-notch fa-spin text-3xl text-[#00D2FF]"></i>
              <p className="text-xs font-mono text-[#00D2FF]">Running MultiBranch 1D-ResNet34 & Territory Mapping...</p>
            </div>
          )}

          {status === 'complete' && analysisResult && (
            <div className="space-y-6">
              <div className="bg-[#0F141C] p-6 rounded-3xl border border-white/10 space-y-4">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-[#00E676] font-bold">ANATOMICAL TERRITORY</span>
                  <span className="px-3 py-1 rounded-full bg-[#FF3D57]/20 text-[#FF3D57] font-bold">{analysisResult.urgency_level || 'ROUTINE'}</span>
                </div>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div className="bg-[#06080D] p-3 rounded-xl">
                    <div className="text-slate-400">Suspected Artery:</div>
                    <div className="font-bold text-white text-sm">{analysisResult.suspected_artery || 'N/A'}</div>
                  </div>
                  <div className="bg-[#06080D] p-3 rounded-xl">
                    <div className="text-slate-400">Predicted Class:</div>
                    <div className="font-bold text-[#00E676] text-sm">{analysisResult.predicted_class} ({((analysisResult.confidence_score || 0.95) * 100).toFixed(1)}%)</div>
                  </div>
                </div>
                <div className="pt-2 flex justify-between items-center border-t border-white/10 text-xs">
                  <span className="text-slate-400">Cath Lab Transfer:</span>
                  <a href={qcaUrl} className="px-5 py-2 rounded-full bg-[#00D2FF]/20 text-[#00D2FF] font-bold hover:bg-[#00D2FF] hover:text-black transition-all uppercase">
                    TRANSFER TO ANGIOGRAPHY QCA SUITE &rarr;
                  </a>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
