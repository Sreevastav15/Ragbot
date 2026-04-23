// src/components/MetricsPanel.jsx
import "./MetricsPanel.css";

function MetricsPanel({ metrics, onClose }) {
  if (!metrics) return null;

  const { responseTimeMs, sources = [] } = metrics;

  return (
    <div className="metrics-panel animate-in">
      <div className="metrics-header">
        <span className="metrics-title">⚡ RAG Metrics</span>
        <button className="metrics-close" onClick={onClose}>✕</button>
      </div>
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-val">{responseTimeMs}ms</span>
          <span className="metric-name">Response Time</span>
        </div>
        <div className="metric-card">
          <span className="metric-val">{sources.length}</span>
          <span className="metric-name">Sources Used</span>
        </div>
      </div>
      {sources.length > 0 && (
        <div className="metrics-sources">
          <p className="metrics-subtitle">Retrieved chunks:</p>
          {sources.map((s, i) => (
            <div key={i} className="metrics-source-row">
              <span className="ms-index">#{i + 1}</span>
              <span className="ms-filename">{s.filename}</span>
              {s.page && <span className="ms-page">p.{s.page}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default MetricsPanel;