import React, { useEffect, useMemo, useState } from 'react';
import './App.css';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  FileSearch,
  Gauge,
  Image as ImageIcon,
  Loader2,
  Play,
  RotateCcw,
  Settings,
  ShieldCheck,
  UploadCloud,
  Video,
  X,
} from 'lucide-react';

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
  { id: 'voting', label: 'Vote' },
  { id: 'average', label: 'Average' },
  { id: 'stacking', label: 'Stack' },
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
  const [showHealth, setShowHealth] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [ensembleMethod, setEnsembleMethod] = useState('voting');

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

  const availableModels = useMemo(() => {
    const media = systemHealth?.media_type_details?.[activeMediaType]?.models || {};
    return Object.entries(media);
  }, [activeMediaType, systemHealth]);

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
  const activeModelCount = availableModels.filter(([, model]) => model.status === 'healthy').length;
  const analyzeLabel = activeMediaType === 'video' ? 'Analyze video' : 'Analyze image';

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
          <div className="ds-kpi">{activeModelCount || 0} checks ready</div>
          <button className={`ds-status ${statusClass}`} onClick={() => setShowHealth(true)}>
            <span />
            {status}
            <ChevronDown size={16} />
          </button>
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
            </div>

            <div className="ds-mode-grid" aria-label="Media type">
              <button className={activeMediaType === 'image' ? 'active' : ''} onClick={() => setActiveMediaType('image')}>
                <ImageIcon size={19} />
                <span>
                  <strong>Image</strong>
                  <small>Photos and generated stills</small>
                </span>
              </button>
              <button className={activeMediaType === 'video' ? 'active' : ''} onClick={() => setActiveMediaType('video')}>
                <Video size={19} />
                <span>
                  <strong>Video</strong>
                  <small>Clips and face-swap media</small>
                </span>
              </button>
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
                  <span>Ensemble</span>
                  <strong>{ensembleMethod}</strong>
                </div>
              </div>

              <div className="ds-action-buttons">
                <button className="ds-run-button" onClick={handleAnalyze} disabled={!selectedFile || isAnalyzing}>
                  {isAnalyzing ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                  {isAnalyzing ? 'Analyzing' : analyzeLabel}
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

            {results ? (
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

      {showHealth && (
        <HealthModal
          activeMediaType={activeMediaType}
          availableModels={availableModels}
          onClose={() => setShowHealth(false)}
          status={status}
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
            <label>Sensitivity threshold</label>
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
            <label>Ensemble method</label>
          </div>
          <div className="ds-segmented">
            {METHODS.map((method) => (
              <button
                key={method.id}
                className={ensembleMethod === method.id ? 'active' : ''}
                onClick={() => setEnsembleMethod(method.id)}
              >
                {method.label}
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

function HealthModal({ activeMediaType, availableModels, onClose, status }) {
  return (
    <div className="ds-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="ds-settings-modal ds-health-modal" role="dialog" aria-modal="true" aria-label="System health" onMouseDown={(event) => event.stopPropagation()}>
        <div className="ds-modal-header">
          <div>
            <span className="ds-label">System</span>
            <h3>System health</h3>
          </div>
          <button className="ds-icon-button" onClick={onClose} aria-label="Close model health">
            <X size={18} />
          </button>
        </div>

        <div className="ds-health-summary">
          <span className={status === 'healthy' ? 'dot green' : 'dot amber'} />
          <strong>{status}</strong>
          <small>{activeMediaType} pipeline</small>
        </div>

        <div className="ds-health-list">
          {availableModels.length === 0 ? (
            <p className="ds-muted">No check details available for {activeMediaType}.</p>
          ) : availableModels.map(([id, model], index) => (
            <div key={id} className="ds-health-item">
              <span className={model.status === 'healthy' ? 'dot green' : 'dot amber'} />
              <div>
                <strong>Check {index + 1}</strong>
                <small>{model.status || 'unknown'} {model.model_loaded ? '/ loaded' : '/ idle'}</small>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function EmptyResults() {
  return (
    <section className="ds-empty-results">
      <FileSearch size={20} />
      <div>
        <strong>Results will appear here</strong>
        <span>Run an analysis to view a simple authenticity report.</span>
      </div>
    </section>
  );
}

function ResultsPanel({ results, mediaType, threshold }) {
  const isFake = results.is_likely_deepfake;
  const probability = results.deepfake_probability || 0;
  const checks = Object.values(results.model_results || {}).filter((result) => !result.error);
  const flaggedChecks = checks.filter((result) => (
    result.class === 'fake' || (typeof result.probability === 'number' && result.probability >= threshold)
  )).length;
  const confidence = Math.abs(probability - threshold);
  const confidenceLabel = confidence > 0.25 ? 'High confidence' : confidence > 0.1 ? 'Moderate confidence' : 'Close call';
  const evidenceTone = isFake
    ? 'The media contains visual patterns that are commonly seen in manipulated or AI-generated content.'
    : 'The media mostly matches patterns expected from authentic, unaltered content.';
  const recommendation = isFake
    ? 'Treat this file as suspicious and verify it with the original source before sharing or relying on it.'
    : 'This file does not show strong signs of manipulation, but important media should still be verified with its source.';

  return (
    <section className="ds-results">
      <div className={`ds-verdict ${isFake ? 'fake' : 'real'}`}>
        <div className="ds-verdict-icon">
          {isFake ? <AlertTriangle size={30} /> : <CheckCircle2 size={30} />}
        </div>
        <div className="ds-verdict-copy">
          <span className="ds-label">Verdict</span>
          <h3>{isFake ? 'Likely manipulated' : `Likely authentic ${mediaType}`}</h3>
          <p>{confidenceLabel} assessment</p>
        </div>
        <div className="ds-probability">
          <span>{pct(probability)}</span>
          <small>AI probability</small>
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
              <span>AI likelihood</span>
              <strong>{pct(probability)}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{confidenceLabel}</strong>
            </div>
            <div>
              <span>Checks completed</span>
              <strong>{checks.length || 'N/A'}</strong>
            </div>
            <div>
              <span>Signals flagged</span>
              <strong>{checks.length ? `${flaggedChecks} of ${checks.length}` : 'N/A'}</strong>
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
