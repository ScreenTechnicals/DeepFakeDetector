import React from 'react';
import { AlertTriangle, CheckCircle2, Download, Eye, Activity, Sun, Mic, Brain } from 'lucide-react';

/**
 * HeatmapOverlay — interactive Grad-CAM heatmap viewer.
 *
 * Shows three view modes: Original / Heatmap / Overlay.
 * Displays the facial region label at the peak activation point.
 */
function HeatmapOverlay({ gradcam, apiBase }) {
  const [viewMode, setViewMode] = React.useState('overlay');

  if (!gradcam) return null;

  const overlayUrl = gradcam.overlay_path
    ? `${apiBase}/api/gradcam/${encodeURIComponent(gradcam.overlay_path)}`
    : null;
  const heatmapUrl = gradcam.heatmap_path
    ? `${apiBase}/api/gradcam/${encodeURIComponent(gradcam.heatmap_path)}`
    : null;

  // For now, display the overlay/heatmap path as-is if served statically
  const displayUrl =
    viewMode === 'overlay' ? overlayUrl :
    viewMode === 'heatmap' ? heatmapUrl : null;

  return (
    <div className="df-heatmap">
      <div className="df-heatmap-controls">
        {['original', 'heatmap', 'overlay'].map((mode) => (
          <button
            key={mode}
            className={viewMode === mode ? 'active' : ''}
            onClick={() => setViewMode(mode)}
          >
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </button>
        ))}
      </div>

      <div className="df-heatmap-canvas">
        {displayUrl ? (
          <img src={displayUrl} alt={`Grad-CAM ${viewMode} view`} />
        ) : (
          <div className="df-heatmap-placeholder">
            <Eye size={32} />
            <span>Heatmap not available for this view</span>
          </div>
        )}
      </div>

      {gradcam.region_label && (
        <div className="df-region-tag">
          <span className="df-region-label">Dominant region</span>
          <strong>{gradcam.region_label.replace('_', ' ')}</strong>
          {gradcam.peak_coordinate && (
            <small>Peak at ({gradcam.peak_coordinate[0]}, {gradcam.peak_coordinate[1]})</small>
          )}
        </div>
      )}
    </div>
  );
}

export default HeatmapOverlay;
