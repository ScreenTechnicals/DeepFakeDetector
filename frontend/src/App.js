import React, { useEffect, useMemo, useState } from 'react';
import './App.css';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  FileSearch,
  Gauge,
  Image as ImageIcon,
  Loader2,
  Play,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
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

const MODELS = {
  npr_deepfakedetection: 'NPR Deepfake',
  universalfakedetect: 'UniversalFakeDetect',
  cross_efficient_vit: 'Cross Efficient ViT',
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

  return (
    <div className="ds-app">
      <header className="ds-topbar">
        <div className="ds-brand">
          <div className="ds-brand-mark">
            <ShieldCheck size={22} />
          </div>
          <div>
            <h1>DeepSafe</h1>
            <p>Media authenticity console</p>
          </div>
        </div>

        <div className="ds-header-actions">
          <div className="ds-kpi">
            <span>{activeMediaType}</span>
            <strong>{activeModelCount || 0} models ready</strong>
          </div>
          <button className={`ds-status ${statusClass}`} onClick={() => setShowHealth((value) => !value)}>
            <span />
            {status}
            <ChevronDown size={16} className={showHealth ? 'rotated' : ''} />
          </button>
        </div>
      </header>

      <main className="ds-shell">
        <section className="ds-titlebar">
          <div>
            <p className="ds-eyebrow">Analysis workspace</p>
            <h2>Inspect media</h2>
          </div>
          <div className="ds-media-toggle" aria-label="Media type">
            <button className={activeMediaType === 'image' ? 'active' : ''} onClick={() => setActiveMediaType('image')}>
              <ImageIcon size={17} /> Image
            </button>
            <button className={activeMediaType === 'video' ? 'active' : ''} onClick={() => setActiveMediaType('video')}>
              <Video size={17} /> Video
            </button>
          </div>
        </section>

        {showHealth && (
          <section className="ds-health-panel">
            <div className="ds-panel-heading">
              <Activity size={18} />
              Model status
            </div>
            <div className="ds-health-grid">
              {availableModels.length === 0 ? (
                <p className="ds-muted">No model details available for {activeMediaType}.</p>
              ) : availableModels.map(([id, model]) => (
                <div key={id} className="ds-health-item">
                  <span className={model.status === 'healthy' ? 'dot green' : 'dot amber'} />
                  <div>
                    <strong>{MODELS[id] || id}</strong>
                    <small>{model.status || 'unknown'} {model.model_loaded ? '/ loaded' : '/ idle'}</small>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="ds-workspace">
          <div className="ds-media-panel">
            <div className="ds-panel-heading">
              <UploadCloud size={18} />
              Source media
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

            <div className="ds-file-strip">
              <div>
                <span className="ds-label">File</span>
                <strong>{selectedFile ? selectedFile.name : 'No file selected'}</strong>
                <small>{selectedFile ? `${formatSize(selectedFile.size)} / ${activeMediaType}` : 'Choose a file or load a sample'}</small>
              </div>
              {selectedFile && (
                <button className="ds-secondary-button" onClick={clearSelection}>
                  <RotateCcw size={16} />
                  Reset
                </button>
              )}
            </div>

            {error && (
              <div className="ds-error">
                <AlertCircle size={18} />
                {error}
              </div>
            )}
          </div>

          <aside className="ds-control-panel">
            <div className="ds-panel-heading">
              <SlidersHorizontal size={18} />
              Analysis setup
            </div>

            <div className="ds-control">
              <div className="ds-control-label">
                <label>Sensitivity</label>
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
                <label>Ensemble</label>
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

            <button className="ds-run-button" onClick={handleAnalyze} disabled={!selectedFile || isAnalyzing}>
              {isAnalyzing ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              {isAnalyzing ? 'Analyzing' : 'Run analysis'}
            </button>

            <div className="ds-control">
              <div className="ds-control-label">
                <label>Samples</label>
              </div>
              <div className="ds-demo-grid">
                {demoItems.map((demo) => (
                  <button key={demo.id} onClick={() => handleDemoMedia(demo)}>
                    <span className={demo.badge === 'real' ? 'badge real' : 'badge fake'}>{demo.badge}</span>
                    {demo.label}
                  </button>
                ))}
              </div>
            </div>
          </aside>
        </section>

        {results ? (
          <ResultsPanel results={results} mediaType={activeMediaType} threshold={threshold} />
        ) : (
          <EmptyResults />
        )}
      </main>
    </div>
  );
}

function EmptyResults() {
  return (
    <section className="ds-empty-results">
      <FileSearch size={20} />
      <div>
        <strong>Results will appear here</strong>
        <span>Run an analysis to view verdict, probability, and model evidence.</span>
      </div>
    </section>
  );
}

function ResultsPanel({ results, mediaType, threshold }) {
  const isFake = results.is_likely_deepfake;
  const probability = results.deepfake_probability || 0;
  const modelResults = Object.entries(results.model_results || {});
  const confidence = Math.abs(probability - threshold);
  const confidenceLabel = confidence > 0.25 ? 'High confidence' : confidence > 0.1 ? 'Moderate confidence' : 'Close call';

  return (
    <section className="ds-results">
      <div className={`ds-verdict ${isFake ? 'fake' : 'real'}`}>
        <div className="ds-verdict-icon">
          {isFake ? <AlertTriangle size={30} /> : <CheckCircle2 size={30} />}
        </div>
        <div className="ds-verdict-copy">
          <span className="ds-label">Verdict</span>
          <h3>{isFake ? 'Likely manipulated' : `Likely authentic ${mediaType}`}</h3>
          <p>{confidenceLabel} / {results.ensemble_method_used || 'voting'} ensemble</p>
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
            Probability scale
          </div>
          <div className="ds-scale">
            <span>Authentic</span>
            <span>Threshold {threshold.toFixed(2)}</span>
            <span>Manipulated</span>
            <div className="ds-scale-track">
              <div className="ds-threshold-pin" style={{ left: `${threshold * 100}%` }} />
              <div className="ds-scale-pin" style={{ left: `${probability * 100}%` }} />
            </div>
          </div>
        </div>

        <div className="ds-score-panel">
          <div className="ds-panel-heading">
            <BarChart3 size={18} />
            Model evidence
          </div>
          <div className="ds-model-list">
            {modelResults.length === 0 ? (
              <p className="ds-muted">No model-level data returned.</p>
            ) : modelResults.map(([name, result]) => (
              <div key={name} className="ds-model-row">
                <div>
                  <strong>{MODELS[name] || name}</strong>
                  <small>{result.error ? result.error : result.class || 'unknown'}</small>
                </div>
                {!result.error && <span>{pct(result.probability)}</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default App;
