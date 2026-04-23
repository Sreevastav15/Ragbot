// src/components/DocumentBadges.jsx
import "./DocumentBadges.css";

function DocumentBadges({ docs, onRemove }) {
  if (!docs.length) return null;

  return (
    <div className="doc-badges-bar">
      <span className="doc-badges-label">Active:</span>
      {docs.map((doc) => (
        <div key={doc.id} className="doc-badge">
          <span className="doc-icon">📄</span>
          <span className="doc-name" title={doc.filename}>{doc.filename}</span>
          {doc.chunkCount != null && (
            <span className="doc-chunks">{doc.chunkCount} chunks</span>
          )}
          <button
            className="doc-remove"
            onClick={() => onRemove(doc.id)}
            aria-label={`Remove ${doc.filename}`}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

export default DocumentBadges;