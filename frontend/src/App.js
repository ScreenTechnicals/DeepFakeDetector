import React, { useEffect, useState, useCallback } from 'react';
import './App.css';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  FileSearch,
  Gauge,
  Info,
  Image as ImageIcon,
  Loader2,
  Microscope,
  Play,
  RotateCcw,
  Settings,
  ShieldCheck,
  UploadCloud,
  Video,
  X,
} from 'lucide-react';
import ForensicResults from './components/ForensicResults';

const API_BASE_URL = window.location.hostname === 'localhost' && window.location.port === '3000'
  ? 'http://localhost:8001'
  : '/api';

const DEMO_MEDIA = {
  image: [
    { id: 'real_portrait', label: 'Real portrait', path: '/demo_images/real_sample.jpg', type: 'image/jpeg', badge: 'real' },
    { id: 'gan_face', label: 'AI face', path: '/demo_images/fake_sample_gan.jpg', type: 'image/jpeg', badge: 'fake' },
    { id: 'stylegan', label: 'AI scene', path: '/demo_images/fake_sample_stylegan.jpg', type: 'image/jpeg', badge: 'fake' },
    { id: 'face_swap', label: 'Face swap', path: '/demo_images/fake_sample_faceswap.jpg', type: 'image/jpeg', badge: 'fake' },
  ],
  video: [
    { id: 'real_video_1', label: 'Real video 1', path: '/demo_videos/real/1.mp4', type: 'video/mp4', badge: 'real' },
    { id: 'real_video_2', label: 'Real video 2', path: '/demo_videos/real/2.mp4', type: 'video/mp4', badge: 'real' },
    { id: 'fake_video_1', label: 'AI video 1', path: '/demo_videos/fake/1.mp4', type: 'video/mp4', badge: 'fake' },
    { id: 'fake_video_2', label: 'AI video 2', path: '/demo_videos/fake/2.mp4', type: 'video/mp4', badge: 'fake' },
  ],
};

const METHODS = [
  { id: 'voting', label: 'Balanced', hint: 'Best default for everyday checks. It weighs the available signals evenly.' },
  { id: 'average', label: 'Score-based', hint: 'Uses probability scores directly. Useful when you want a smoother result.' },
  { id: 'stacking', label: 'Strict', hint: 'Uses the strictest review path when available. It may flag more borderline files.' },
];

function inferMediaType(file) {
  if (!file) return 'image';
  if (file.type.startsWith('video/')) return 'video';
  if (/\.(mp4|mov|avi|mkv|m4v)$/i.test(file.name)) return 'video';
  return 'image';
}

function formatSize(bytes = 0) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / 1024).toFixed(2)} KB`;
}

function pct(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '0.0%';
  return `${(Math.max(0, Math.min(value, 1)) * 100).toFixed(1)}%`;
}

function App() {
  const [activeMediaType, setActiveMediaType] = useState('image');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [ensembleMethod, setEnsembleMethod] = useState('voting');

  // Forensic analysis mode state
  const [forensicMode, setForensicMode] = useState(false);
  const [forensicJobId, setForensicJobId] = useState(null);
  const [forensicResult, setForensicResult] = useState(null);
  const [forensicStatus, setForensicStatus] = useState(null); // pending, processing, complete, failed

  const handleForensicAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);
    setResults(null);
    setForensicResult(null);
    setForensicStatus('pending');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Forensic analysis submission failed');
      }
      const { job_id } = await res.json();
      setForensicJobId(job_id);

      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE_URL}/api/status/${job_id}`);
          const statusData = await statusRes.json();
          setForensicStatus(statusData.status);

          if (statusData.status === 'complete') {
            clearInterval(poll);
            setForensicResult(statusData.result);
            setIsAnalyzing(false);
          } else if (statusData.status === 'failed') {
            clearInterval(poll);
            setError(statusData.error || 'Forensic analysis failed');
            setIsAnalyzing(false);
          }
        } catch {
          // polling error — keep trying
        }
      }, 1000);
    } catch (err) {
      setError(err.message);
      setIsAnalyzing(false);
    }
  }, [selectedFile]);

  useEffect(() => {
    let ignore = false;
    fetch(`${API_BASE_URL}/health`)
      .then((res) => res.json())
      .then((data) => {
        if (!ignore) setSystemHealth(data);
      })
      .catch(() => {
        if (!ignore) setSystemHealth({ overall_api_status: 'unreachable' });
      });
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFile = (file) => {
    if (!file) {
      setSelectedFile(null);
      setPreviewUrl(null);
      setResults(null);
      setError(null);
      return;
    }
    const nextMediaType = inferMediaType(file);
    setActiveMediaType(nextMediaType);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
  };

  const handleInputChange = (event) => {
    handleFile(event.target.files?.[0] || null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    if (!isAnalyzing) handleFile(event.dataTransfer.files?.[0] || null);
  };

  const handleDemoMedia = async (demo) => {
    try {
      const response = await fetch(demo.path);
      if (!response.ok) throw new Error(`Unable to load ${demo.label}`);
      const blob = await response.blob();
      const file = new File([blob], demo.path.split('/').pop(), { type: demo.type });
      handleFile(file);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', String(threshold));
    formData.append('ensemble_method', ensembleMethod);

    try {
      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
      }
      setResults(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearSelection = (event) => {
    event?.preventDefault();
    handleFile(null);
  };

  const status = systemHealth?.overall_api_status || 'checking';
  const statusClass = status === 'healthy' ? 'is-healthy' : 'is-warning';
  const demoItems = DEMO_MEDIA[activeMediaType] || [];
  const analyzeLabel = activeMediaType === 'video' ? 'Analyze video' : 'Analyze image';
  const reviewStyleLabel = METHODS.find((method) => method.id === ensembleMethod)?.label || 'Balanced';

  return (
    <div className="ds-app">
      <header className="ds-topbar">
        <div className="ds-brand">
          <div className="ds-brand-mark">
            <ShieldCheck size={22} />
          </div>
          <div>
            <h1>DeepSafe</h1>
            <p>Authenticity analysis</p>
          </div>
        </div>

        <div className="ds-header-title">
          <span>Analyze</span>
          <strong>Image and video inspection</strong>
        </div>

        <div className="ds-header-actions">
          <div className={`ds-status ${statusClass}`}>
            <span />
            {status}
          </div>
        </div>
      </header>

      <main className="ds-shell">
        <section className="ds-workspace">
          <section className="ds-media-panel">
            <div className="ds-panel-top">
              <div>
                <p className="ds-eyebrow">Input</p>
                <h2>Select media</h2>
              </div>

              <div className="ds-mode-grid" aria-label="Media type">
                <button className={activeMediaType === 'image' && !forensicMode ? 'active' : ''} onClick={() => { setActiveMediaType('image'); setForensicMode(false); }}>
                  <ImageIcon size={19} />
                  <span>
                    <strong>Image</strong>
                    <small>Photos and generated stills</small>
                  </span>
                </button>
                <button className={activeMediaType === 'video' && !forensicMode ? 'active' : ''} onClick={() => { setActiveMediaType('video'); setForensicMode(false); }}>
                  <Video size={19} />
                  <span>
                    <strong>Video</strong>
                    <small>Clips and face-swap media</small>
                  </span>
                </button>
                <button className={forensicMode ? 'active forensic' : ''} onClick={() => { setForensicMode(true); setActiveMediaType('video'); }}>
                  <Microscope size={19} />
                  <span>
                    <strong>Forensic</strong>
                    <small>Full explainable analysis</small>
                  </span>
                </button>
              </div>
            </div>

            <label
              className={`ds-dropzone ${dragActive ? 'active' : ''} ${selectedFile ? 'has-preview' : ''}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept={activeMediaType === 'video' ? 'video/*' : 'image/*'}
                onChange={handleInputChange}
                disabled={isAnalyzing}
              />
              {previewUrl ? (
                <div className="ds-preview">
                  <button type="button" className="ds-icon-button ds-clear" onClick={clearSelection} aria-label="Clear selected media">
                    <X size={17} />
                  </button>
                  {activeMediaType === 'video' ? (
                    <video src={previewUrl} controls />
                  ) : (
                    <img src={previewUrl} alt="Selected media preview" />
                  )}
                </div>
              ) : (
                <div className="ds-drop-empty">
                  <div className="ds-upload-icon">
                    <UploadCloud size={32} />
                  </div>
                  <strong>Drop a {activeMediaType} file here</strong>
                  <span>or click to browse</span>
                </div>
              )}
            </label>

            {error && (
              <div className="ds-error">
                <AlertCircle size={18} />
                {error}
              </div>
            )}

            <div className="ds-command-bar">
              <div className="ds-action-summary">
                <span className="ds-label">Selected file</span>
                <strong>{selectedFile ? selectedFile.name : 'Nothing selected yet'}</strong>
                <small>{selectedFile ? `${formatSize(selectedFile.size)} / ${activeMediaType}` : 'Upload media or choose a sample from settings'}</small>
              </div>

              <div className="ds-analysis-meta">
                <div>
                  <span>Threshold</span>
                  <strong>{threshold.toFixed(2)}</strong>
                </div>
                <div>
                <span>Review style</span>
                <strong>{reviewStyleLabel}</strong>
                </div>
              </div>

              <div className="ds-action-buttons">
                <button className="ds-run-button" onClick={forensicMode ? handleForensicAnalyze : handleAnalyze} disabled={!selectedFile || isAnalyzing}>
                  {isAnalyzing ? <Loader2 className="spin" size={18} /> : forensicMode ? <Microscope size={18} /> : <Play size={18} />}
                  {isAnalyzing ? (forensicMode ? `Analyzing (${forensicStatus || 'pending'})` : 'Analyzing') : (forensicMode ? 'Run forensic analysis' : analyzeLabel)}
                </button>
                <button className="ds-secondary-button" onClick={() => setShowSettings(true)}>
                  <Settings size={16} />
                  Settings
                </button>
                {selectedFile && (
                  <button className="ds-secondary-button" onClick={clearSelection}>
                    <RotateCcw size={16} />
                    Reset
                  </button>
                )}
              </div>
            </div>
          </section>

          <section className="ds-output-panel">
            <div className="ds-panel-top">
              <div>
                <p className="ds-eyebrow">Output</p>
                <h2>Analysis result</h2>
              </div>
            </div>

            {forensicMode && forensicResult ? (
              <ForensicResults result={forensicResult} jobId={forensicJobId} apiBase={API_BASE_URL} />
            ) : results ? (
              <ResultsPanel results={results} mediaType={activeMediaType} threshold={threshold} />
            ) : (
              <EmptyResults />
            )}
          </section>
        </section>
      </main>

      {showSettings && (
        <SettingsModal
          activeMediaType={activeMediaType}
          demoItems={demoItems}
          ensembleMethod={ensembleMethod}
          onClose={() => setShowSettings(false)}
          onDemoMedia={handleDemoMedia}
          setEnsembleMethod={setEnsembleMethod}
          setThreshold={setThreshold}
          threshold={threshold}
        />
      )}

    </div>
  );
}

function SettingsModal({
  activeMediaType,
  demoItems,
  ensembleMethod,
  onClose,
  onDemoMedia,
  setEnsembleMethod,
  setThreshold,
  threshold,
}) {
  return (
    <div className="ds-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="ds-settings-modal" role="dialog" aria-modal="true" aria-label="Analysis settings" onMouseDown={(event) => event.stopPropagation()}>
        <div className="ds-modal-header">
          <div>
            <span className="ds-label">Settings</span>
            <h3>Analysis controls</h3>
          </div>
          <button className="ds-icon-button" onClick={onClose} aria-label="Close settings">
            <X size={18} />
          </button>
        </div>

        <div className="ds-control">
          <div className="ds-control-label">
            <div className="ds-label-with-help">
              <label>Sensitivity threshold</label>
              <span className="ds-help" tabIndex="0" aria-label="Higher sensitivity flags more files as suspicious. Lower sensitivity reduces false positives.">
                <Info size={15} />
                <span className="ds-tooltip">Higher values make DeepSafe stricter. Lower values reduce false positives.</span>
              </span>
            </div>
            <strong>{threshold.toFixed(2)}</strong>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.01"
            value={threshold}
            onChange={(event) => setThreshold(parseFloat(event.target.value))}
          />
          <div className="ds-control-row">
            <span>Lower false positives</span>
            <span>Stricter flagging</span>
          </div>
        </div>

        <div className="ds-control">
          <div className="ds-control-label">
            <div className="ds-label-with-help">
              <label>Review style</label>
              <span className="ds-help" tabIndex="0" aria-label="Choose how DeepSafe combines available evidence.">
                <Info size={15} />
                <span className="ds-tooltip">Controls how DeepSafe combines the available evidence into one report.</span>
              </span>
            </div>
          </div>
          <div className="ds-segmented">
            {METHODS.map((method) => (
              <button
                key={method.id}
                className={ensembleMethod === method.id ? 'active' : ''}
                onClick={() => setEnsembleMethod(method.id)}
                title={method.hint}
                aria-label={`${method.label}. ${method.hint}`}
              >
                {method.label}
                <span className="ds-segmented-tooltip">{method.hint}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="ds-control">
          <div className="ds-control-label">
            <label>{activeMediaType} samples</label>
          </div>
          <div className="ds-demo-grid">
            {demoItems.map((demo) => (
              <button
                key={demo.id}
                onClick={() => {
                  onDemoMedia(demo);
                  onClose();
                }}
              >
                <span className={demo.badge === 'real' ? 'badge real' : 'badge fake'}>{demo.badge}</span>
                {demo.label}
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function EmptyResults() {
  return (
    <section className="ds-empty-results">
      <div className="ds-empty-icon">
        <FileSearch size={19} />
      </div>
      <div>
        <strong>Awaiting analysis</strong>
        <span>Upload media, then run DeepSafe to generate a plain-language report.</span>
      </div>
    </section>
  );
}

function ResultsPanel({ results, mediaType, threshold }) {
  const modelResults = Object.values(results.model_results || {});
  const failedChecks = modelResults.filter((result) => result.error);
  const expectedChecks = modelResults.length || results.model_count || 0;
  const checksCompleted = results.model_count || modelResults.filter((result) => !result.error).length;
  const isIncomplete = failedChecks.length > 0 || (expectedChecks > 0 && checksCompleted < expectedChecks);
  const isFake = !isIncomplete && results.is_likely_deepfake;
  const probability = typeof results.deepfake_probability === 'number' ? results.deepfake_probability : 0;
  const methodUsed = results.ensemble_method_used || results.ensemble_method_requested || 'default';
  const confidence = Math.abs(probability - threshold);
  const confidenceLabel = confidence > 0.25 ? 'High confidence' : confidence > 0.1 ? 'Moderate confidence' : 'Close call';
  const verdictTitle = isIncomplete
    ? 'Analysis incomplete'
    : isFake
      ? 'Likely manipulated'
      : `Likely authentic ${mediaType}`;
  const verdictDetail = isIncomplete
    ? 'Some checks could not finish'
    : `${confidenceLabel} assessment`;
  const evidenceTone = isIncomplete
    ? `DeepSafe completed ${checksCompleted} of ${expectedChecks || checksCompleted} checks. The final verdict is paused because one detector did not return a result.`
    : isFake
    ? `DeepSafe returned a fake likelihood of ${pct(probability)}, which is above the current ${threshold.toFixed(2)} decision threshold.`
    : `DeepSafe returned a fake likelihood of ${pct(probability)}, which is below the current ${threshold.toFixed(2)} decision threshold.`;
  const recommendation = isIncomplete
    ? 'Run the analysis again after the unavailable detector is healthy. Do not treat this file as authentic from this partial result.'
    : isFake
    ? 'Treat this file as suspicious and verify it with the original source before sharing or relying on it.'
    : 'This file does not show strong signs of manipulation, but important media should still be verified with its source.';

  return (
    <section className="ds-results">
      <div className={`ds-verdict ${isIncomplete ? 'warning' : isFake ? 'fake' : 'real'}`}>
        <div className="ds-verdict-icon">
          {isIncomplete ? <AlertCircle size={30} /> : isFake ? <AlertTriangle size={30} /> : <CheckCircle2 size={30} />}
        </div>
        <div className="ds-verdict-copy">
          <span className="ds-label">Verdict</span>
          <h3>{verdictTitle}</h3>
          <p>{verdictDetail}</p>
        </div>
        <div className="ds-probability">
          <span>{isIncomplete ? '--' : pct(probability)}</span>
          <small>{isIncomplete ? 'final score paused' : 'AI probability'}</small>
        </div>
      </div>

      <div className="ds-results-grid">
        <div className="ds-score-panel">
          <div className="ds-panel-heading">
            <Gauge size={18} />
            Authenticity report
          </div>
          <div className="ds-report-grid">
            <div>
              <span>Fake likelihood</span>
              <strong>{pct(probability)}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{isIncomplete ? 'Incomplete' : confidenceLabel}</strong>
            </div>
            <div>
              <span>Checks completed</span>
              <strong>{expectedChecks ? `${checksCompleted} / ${expectedChecks}` : checksCompleted || 'N/A'}</strong>
            </div>
            <div>
              <span>{isIncomplete ? 'Checks unavailable' : 'Review method'}</span>
              <strong>{isIncomplete ? failedChecks.length : methodUsed}</strong>
            </div>
          </div>
        </div>

        <div className="ds-score-panel">
          <div className="ds-panel-heading">
            <FileSearch size={18} />
            Evidence summary
          </div>
          <div className="ds-report-copy">
            <p>{evidenceTone}</p>
            <p>{recommendation}</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default App;
