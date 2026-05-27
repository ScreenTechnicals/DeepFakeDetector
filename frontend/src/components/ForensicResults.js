import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Activity,
  Eye,
  Mic,
  Sun,
  Brain,
  FileText,
  Loader2,
} from 'lucide-react';
import HeatmapOverlay from './HeatmapOverlay';

/**
 * ForensicResults — full forensic analysis panel with semantic
 * score cards, Grad-CAM heatmap, and PDF download.
 *
 * Maps to the AnalysisResult schema from the DeepForensics pipeline.
 */
function ForensicResults({ result, jobId, apiBase }) {
  if (!result) return null;

  const isFake = result.verdict === 'FAKE';
  const confidence = result.confidence || 0;
  const pct = (v) => `${(Math.max(0, Math.min(v, 1)) * 100).toFixed(1)}%`;

  return (
    <section className="df-results">
      {/* Verdict Banner */}
      <div className={`df-verdict ${isFake ? 'fake' : 'real'}`}>
        <div className="df-verdict-icon">
          {isFake ? <AlertTriangle size={28} /> : <CheckCircle2 size={28} />}
        </div>
        <div className="df-verdict-text">
          <span className="df-label">Forensic Verdict</span>
          <h3>{result.verdict}</h3>
          <p>{pct(confidence)} confidence</p>
        </div>
        <div className="df-verdict-score">
          <span>{pct(confidence)}</span>
          <small>calibrated</small>
        </div>
      </div>

      {/* Evidence Summary */}
      {result.summary && (
        <div className="df-summary-card">
          <div className="df-card-heading">
            <FileText size={16} />
            Evidence Summary
          </div>
          <p>{result.summary}</p>
        </div>
      )}

      {/* Semantic Score Cards */}
      <div className="df-scores-grid">
        {/* CNN Backbone Score */}
        <ScoreCard
          icon={<Brain size={18} />}
          title="CNN Backbone"
          subtitle="EfficientNet-B4"
          value={result.visual_only_confidence}
          format={pct}
          status={result.visual_only_confidence > 0.5 ? 'warning' : 'good'}
        />

        {/* Lip-Sync Score */}
        {result.semantic_scores?.lip_sync ? (
          <ScoreCard
            icon={<Mic size={18} />}
            title="Lip-Audio Coherence"
            subtitle="Module 4A"
            items={[
              { label: 'Mean Correlation', value: result.semantic_scores.lip_sync.mean_correlation?.toFixed(3) },
              { label: 'Below Threshold', value: pct(result.semantic_scores.lip_sync.below_threshold_fraction) },
            ]}
            status={result.semantic_scores.lip_sync.mean_correlation > 0.6 ? 'good' : 'warning'}
          />
        ) : (
          <ScoreCard icon={<Mic size={18} />} title="Lip-Audio Coherence" subtitle="No audio available" disabled />
        )}

        {/* Blink Score */}
        {result.semantic_scores?.blink ? (
          <ScoreCard
            icon={<Eye size={18} />}
            title="Blink Dynamics"
            subtitle="Module 4B"
            items={[
              { label: 'Rate', value: `${result.semantic_scores.blink.blink_rate_per_min?.toFixed(1)}/min` },
              { label: 'Z-Score', value: result.semantic_scores.blink.anomaly_z_score?.toFixed(2) },
            ]}
            status={Math.abs(result.semantic_scores.blink.anomaly_z_score) < 1.5 ? 'good' : 'warning'}
          />
        ) : (
          <ScoreCard icon={<Eye size={18} />} title="Blink Dynamics" subtitle="Single image — skipped" disabled />
        )}

        {/* Lighting Score */}
        {result.semantic_scores?.lighting ? (
          <ScoreCard
            icon={<Sun size={18} />}
            title="Lighting Consistency"
            subtitle="Module 4C"
            items={[
              { label: 'Max Angle', value: `${result.semantic_scores.lighting.max_pairwise_angle_deg?.toFixed(1)}°` },
              { label: 'Consistency', value: result.semantic_scores.lighting.consistency_score?.toFixed(3) },
            ]}
            status={result.semantic_scores.lighting.consistency_score > 0.7 ? 'good' : 'warning'}
          />
        ) : (
          <ScoreCard icon={<Sun size={18} />} title="Lighting Consistency" subtitle="Unavailable" disabled />
        )}
      </div>

      {/* Grad-CAM Heatmap */}
      <div className="df-section">
        <div className="df-card-heading">
          <Activity size={16} />
          Grad-CAM Heatmap
        </div>
        <HeatmapOverlay gradcam={result.gradcam} apiBase={apiBase} />
      </div>

      {/* Audit Block */}
      {result.audit && (
        <div className="df-audit">
          <div className="df-card-heading">
            <FileText size={16} />
            Audit Record
          </div>
          <div className="df-audit-grid">
            <div><span>Timestamp</span><strong>{result.audit.timestamp}</strong></div>
            <div><span>Input Hash</span><strong>{result.audit.input_file_sha256 || 'N/A'}</strong></div>
            <div><span>Total Time</span><strong>{result.audit.total_time?.toFixed(2)}s</strong></div>
            {result.audit.module_times && Object.entries(result.audit.module_times).map(([mod, t]) => (
              <div key={mod}><span>{mod}</span><strong>{t.toFixed(3)}s</strong></div>
            ))}
          </div>
        </div>
      )}

      {/* PDF Download */}
      {jobId && (
        <div className="df-actions">
          <a
            href={`${apiBase}/api/report/${jobId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="df-download-btn"
          >
            <Download size={16} />
            Download Forensic PDF Report
          </a>
        </div>
      )}
    </section>
  );
}


/**
 * ScoreCard — reusable semantic score card with status indicator.
 */
function ScoreCard({ icon, title, subtitle, value, format, items, status, disabled }) {
  const statusClass = disabled ? 'disabled' : status === 'good' ? 'good' : 'warning';

  return (
    <div className={`df-score-card ${statusClass}`}>
      <div className="df-score-header">
        <div className="df-score-icon">{icon}</div>
        <div>
          <strong>{title}</strong>
          <small>{subtitle}</small>
        </div>
      </div>
      {!disabled && (
        <div className="df-score-body">
          {value !== undefined && format && (
            <div className="df-score-value">{format(value)}</div>
          )}
          {items && items.map((item, i) => (
            <div key={i} className="df-score-row">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      )}
      <div className={`df-score-indicator ${statusClass}`} />
    </div>
  );
}

export default ForensicResults;
