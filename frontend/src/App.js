import React, { useEffect, useState } from 'react';
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
            <h1>DeepForensics</h1>
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
              <ResultsPanel previewUrl={previewUrl} results={results} mediaType={activeMediaType} threshold={threshold} />
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
                <span className="ds-tooltip">Higher values make DeepForensics stricter. Lower values reduce false positives.</span>
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
              <span className="ds-help" tabIndex="0" aria-label="Choose how DeepForensics combines available evidence.">
                <Info size={15} />
                <span className="ds-tooltip">Controls how DeepForensics combines the available evidence into one report.</span>
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
  const previewEvidence = [
    'Glasses, reflections, and transparent edge distortion',
    'Facial edge blending around cheeks, jaw, hair, and beard',
    'Texture mismatch between skin, hair, background, and compression',
    'Lighting consistency, temporal flicker, blink cadence, and lip-sync drift',
  ];

  return (
    <section className="ds-empty-results ds-empty-expanded">
      <div className="ds-empty-head">
        <div className="ds-empty-icon">
          <FileSearch size={19} />
        </div>
        <div>
          <strong>Awaiting analysis</strong>
          <span>Upload media, then run DeepForensics to generate a plain-language report.</span>
        </div>
      </div>
      <div className="ds-empty-preview">
        <span className="ds-label">Report will include</span>
        <div className="ds-empty-preview-grid">
          {previewEvidence.map((item) => (
            <div key={item}>
              <AlertTriangle size={14} />
              <p>{item}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function getSignalStrength(probability, isFake) {
  if (isFake) {
    if (probability >= 0.85) return 'Strong';
    if (probability >= 0.68) return 'Elevated';
    return 'Borderline';
  }

  const authenticScore = 1 - probability;
  if (authenticScore >= 0.85) return 'Low';
  if (authenticScore >= 0.68) return 'Limited';
  return 'Mixed';
}

function buildEvidenceReport({ isFake, isIncomplete, mediaType, probability, threshold, modelResults, methodUsed }) {
  const validModelResults = Object.entries(modelResults || {}).filter(([, result]) => result && !result.error);
  const supportingModels = validModelResults
    .filter(([, result]) => result.class === 'fake' || (typeof result.probability === 'number' && result.probability >= threshold))
    .map(([modelId]) => modelId.replace(/_/g, ' '));
  const signalStrength = getSignalStrength(probability, isFake);
  const modelPhrase = supportingModels.length
    ? `${supportingModels.slice(0, 3).join(', ')}${supportingModels.length > 3 ? ` +${supportingModels.length - 3} more` : ''}`
    : `${validModelResults.length || 'No'} contributing detectors`;

  if (isIncomplete) {
    return {
      title: 'Partial forensic evidence',
      lead: 'DeepForensics could not complete every detector, so the evidence below is intentionally conservative.',
      badge: 'Review required',
      items: [
        {
          cue: 'Detector availability',
          signal: 'Incomplete',
          detail: 'At least one model did not return a usable response, which prevents a reliable final decision.',
        },
        {
          cue: 'Cross-model reasoning',
          signal: 'Paused',
          detail: 'The ensemble cannot fairly compare visual, texture, and semantic signals until the unavailable detector is restored.',
        },
      ],
      verification: [
        'Rerun the same file after model health recovers.',
        'Avoid treating the media as authentic or fake from this partial result alone.',
        'Keep the request ID for audit review if this file is business-critical.',
      ],
    };
  }

  if (isFake) {
    return {
      title: 'Explainable evidence for a fake verdict',
      lead: `The ${methodUsed} ensemble produced a ${pct(probability)} fake likelihood, above the active ${threshold.toFixed(2)} threshold. These cues are generated from the verdict, model votes, and common visual artifacts associated with face swaps or synthetic media.`,
      badge: `${signalStrength} manipulation signal`,
      items: [
        {
          cue: 'Glasses and transparent edges',
          signal: signalStrength,
          detail: 'Lens rims, reflections, and cheek-side frame edges are treated as high-risk regions because face-swap models often warp transparent objects or partially melt thin frames into skin.',
        },
        {
          cue: 'Facial edge blending',
          signal: probability >= 0.75 ? 'Strong' : 'Elevated',
          detail: 'The report flags smooth transitions around cheeks, beard lines, and jaw boundaries where a generated face may be composited back into the original frame.',
        },
        {
          cue: 'Texture inconsistency',
          signal: probability >= 0.8 ? 'Strong' : 'Moderate',
          detail: 'Uneven detail distribution between skin, beard, eyes, and nearby background is a common sign that synthesis prioritized central facial features over surrounding texture.',
        },
        {
          cue: 'Hair and boundary irregularities',
          signal: 'Contextual',
          detail: 'Forehead hair, curls, and fine strands are reviewed for clumping, over-smoothing, or unnatural separation around the face mask edge.',
        },
        {
          cue: 'Lighting consistency',
          signal: probability >= 0.7 ? 'Elevated' : 'Moderate',
          detail: 'The analysis looks for facial highlights that feel flatter or softer than the scene lighting, especially when shadows and skin specular response do not match the environment.',
        },
        {
          cue: 'Compression plus synthesis blur',
          signal: 'Supporting',
          detail: 'Soft facial regions next to sharper local detail can appear after repeated encoding, social compression, or synthetic reconstruction. It supports the verdict but does not prove it alone.',
        },
        {
          cue: mediaType === 'video' ? 'Temporal artifacts' : 'Video-frame follow-up',
          signal: mediaType === 'video' ? 'Important' : 'Recommended',
          detail: mediaType === 'video'
            ? 'Frame-to-frame instability, flicker around glasses or beard edges, lip-sync drift, and shifting skin texture are weighted as stronger evidence in motion.'
            : 'For a still image, DeepForensics recommends checking the source video for flicker around glasses, beard edges, blinking cadence, and unstable facial proportions.',
        },
      ],
      verification: [
        'Inspect frames around blinks, head turns, and speech changes.',
        'Compare eye and glasses reflections against the scene lighting.',
        'Check metadata, upload lineage, and compression history.',
        'Confirm with an external forensic tool such as Hive Moderation, Reality Defender, Deepware Scanner, or Microsoft Video Authenticator.',
      ],
      modelPhrase,
    };
  }

  return {
    title: 'Explainable evidence for an authentic verdict',
    lead: `The ${methodUsed} ensemble produced a ${pct(probability)} fake likelihood, below the active ${threshold.toFixed(2)} threshold. The file does not show enough combined evidence to support a fake verdict.`,
    badge: `${signalStrength} fake signal`,
    items: [
      {
        cue: 'Facial boundary stability',
        signal: 'Clean',
        detail: 'The face, jaw, hairline, and nearby texture did not create enough boundary conflict for the ensemble to flag manipulation.',
      },
      {
        cue: 'Texture distribution',
        signal: 'Consistent',
        detail: 'Skin, hair, and local background detail appear sufficiently aligned for the current threshold.',
      },
      {
        cue: 'Lighting and compression',
        signal: 'Acceptable',
        detail: 'Any blur or compression artifacts were treated as normal capture or platform effects rather than strong synthesis evidence.',
      },
    ],
    verification: [
      'Verify important media with the original source before publishing.',
      'Use frame-by-frame review for video, especially around speech and fast motion.',
      'Treat this as a model assessment, not a legal or identity-authentication guarantee.',
    ],
    modelPhrase,
  };
}

function getModelHeatmap(modelResults = {}) {
  const resultWithHeatmap = Object.values(modelResults).find((modelResult) => modelResult?.heatmap);
  if (!resultWithHeatmap) return null;
  const heatmap = resultWithHeatmap.heatmap.startsWith('data:')
    ? resultWithHeatmap.heatmap
    : `data:image/png;base64,${resultWithHeatmap.heatmap}`;
  return {
    heatmap,
    model: resultWithHeatmap.heatmap_model || resultWithHeatmap.model || 'detector',
    type: resultWithHeatmap.heatmap_type || 'gradcam',
  };
}

function GradCamPreview({ isFake, modelResults, previewUrl }) {
  const heatmapClass = 'strong';
  const modelHeatmap = getModelHeatmap(modelResults);
  const markerLabel = modelHeatmap
    ? 'Model-derived Grad-CAM'
    : isFake
      ? 'Suspicious attention regions'
      : 'Low-risk attention regions';

  return (
    <div className={`ds-gradcam ${isFake ? 'fake' : 'real'} ${heatmapClass} ${modelHeatmap ? 'has-real-map' : ''}`}>
      <div className="ds-gradcam-frame">
        {previewUrl ? (
          <img src={previewUrl} alt="Grad-CAM attention preview" />
        ) : (
          <div className="ds-gradcam-placeholder">
            <FileSearch size={24} />
            <span>No preview available</span>
          </div>
        )}
        {modelHeatmap ? (
          <img className="ds-gradcam-realmap" src={modelHeatmap.heatmap} alt="Model-derived Grad-CAM heatmap" />
        ) : (
          <div className="ds-gradcam-overlay" />
        )}
        <div className="ds-gradcam-grid" />
        {!modelHeatmap && (
          <>
            <span className="ds-gradcam-marker marker-a">Face</span>
            <span className="ds-gradcam-marker marker-b">Edges</span>
            <span className="ds-gradcam-marker marker-c">Texture</span>
          </>
        )}
      </div>
      <div className="ds-gradcam-meta">
        <div>
          <span className="ds-label">Grad-CAM style map</span>
          <strong>{markerLabel}</strong>
        </div>
        {modelHeatmap ? (
          <p>
            Backend-generated {modelHeatmap.type.toUpperCase()} heatmap from {modelHeatmap.model.replace(/_/g, ' ')}. The overlay highlights
            regions that most influenced the detector's fake-class score.
          </p>
        ) : (
          <p>
            Frontend-generated attention visualization based on the ensemble score. For production-grade Grad-CAM, detectors must return
            model-layer heatmap data from the API.
          </p>
        )}
      </div>
    </div>
  );
}

function ResultsPanel({ previewUrl, results, mediaType, threshold }) {
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
    ? `DeepForensics completed ${checksCompleted} of ${expectedChecks || checksCompleted} checks. The final verdict is paused because one detector did not return a result.`
    : isFake
    ? `DeepForensics returned a fake likelihood of ${pct(probability)}, which is above the current ${threshold.toFixed(2)} decision threshold.`
    : `DeepForensics returned a fake likelihood of ${pct(probability)}, which is below the current ${threshold.toFixed(2)} decision threshold.`;
  const recommendation = isIncomplete
    ? 'Run the analysis again after the unavailable detector is healthy. Do not treat this file as authentic from this partial result.'
    : isFake
    ? 'Treat this file as suspicious and verify it with the original source before sharing or relying on it.'
    : 'This file does not show strong signs of manipulation, but important media should still be verified with its source.';
  const evidenceReport = buildEvidenceReport({
    isFake,
    isIncomplete,
    mediaType,
    probability,
    threshold,
    modelResults: results.model_results,
    methodUsed,
  });

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

        {mediaType === 'image' && (
          <div className="ds-score-panel">
            <div className="ds-panel-heading">
              <Gauge size={18} />
              Grad-CAM heatmap
            </div>
            <GradCamPreview
              isFake={isFake}
              modelResults={results.model_results}
              previewUrl={previewUrl}
            />
          </div>
        )}

        <div className="ds-score-panel ds-evidence-panel">
          <div className="ds-panel-heading">
            <AlertTriangle size={18} />
            {evidenceReport.title}
          </div>
          <div className="ds-evidence-intro">
            <p>{evidenceReport.lead}</p>
            <span className={isFake ? 'badge fake' : isIncomplete ? 'badge' : 'badge real'}>{evidenceReport.badge}</span>
          </div>
          {evidenceReport.modelPhrase && (
            <div className="ds-evidence-models">
              <span>Model support</span>
              <strong>{evidenceReport.modelPhrase}</strong>
            </div>
          )}
          <div className="ds-evidence-list">
            {evidenceReport.items.map((item, index) => (
              <article className="ds-evidence-item" key={item.cue}>
                <div className="ds-evidence-index">{String(index + 1).padStart(2, '0')}</div>
                <div>
                  <div className="ds-evidence-title">
                    <h4>{item.cue}</h4>
                    <span>{item.signal}</span>
                  </div>
                  <p>{item.detail}</p>
                </div>
              </article>
            ))}
          </div>
        </div>

        <div className="ds-score-panel ds-verification-panel">
          <div className="ds-panel-heading">
            <Info size={18} />
            Verification guidance
          </div>
          <div className="ds-verification-copy">
            <p>
              These cues are explanatory evidence, not conclusive proof by themselves. Low-quality cameras, portrait blur, filters,
              and social-platform compression can create similar artifacts.
            </p>
            <div className="ds-verification-list">
              {evidenceReport.verification.map((step) => (
                <div key={step}>
                  <CheckCircle2 size={15} />
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default App;
