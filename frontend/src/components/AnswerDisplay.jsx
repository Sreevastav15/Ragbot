// src/components/AnswerDisplay.jsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./AnswerDisplay.css";

function AnswerDisplay({
  question,
  answer,
  sources = [],
  metrics = null,
  responseTimeMs = null,
  rewrittenQuery = null,
  isStreaming = false,
}) {
  if (!question) return null;

  const uniqueSources = Array.from(
    new Map(sources.map((s) => [`${s.filename}:${s.page}`, s])).values()
  );

  return (
    <div className="answer-block animate-in">
      {/* User bubble */}
      <div className="user-row">
        <div className="user-bubble">
          <p>{question}</p>
        </div>
      </div>

      {/* Bot response */}
      <div className="bot-row">
        <div className="bot-avatar">◈</div>
        <div className="bot-content">
          {/* Rewritten query chip */}
          {rewrittenQuery && rewrittenQuery !== question && (
            <div className="rewrite-chip">
              <span className="rewrite-label">Searched for:</span>
              <span className="rewrite-text">{rewrittenQuery}</span>
            </div>
          )}

          {/* Answer body */}
          <div className="bot-bubble">
            {answer ? (
              <>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                {isStreaming && <span className="streaming-cursor" />}
              </>
            ) : (
              <div className="thinking-dots">
                <span /><span /><span />
              </div>
            )}
          </div>

          {/* Footer row */}
          {!isStreaming && (answer || sources.length > 0) && (
            <div className="answer-footer">
              {/* Timing */}
              {responseTimeMs != null && (
                <span className="timing-badge">⚡ {responseTimeMs}ms</span>
              )}

              {/* Sources */}
              {uniqueSources.length > 0 && (
                <div className="sources-row">
                  <span className="sources-label">Sources:</span>
                  {uniqueSources.map((s, i) => (
                    <span key={i} className="source-chip">
                      📄 {s.filename}{s.page ? `, p.${s.page}` : ""}
                    </span>
                  ))}
                </div>
              )}

              {/* Inline metrics */}
              {metrics && (
                <div className="inline-metrics">
                  {metrics.rouge != null && (
                    <span className="metric-tag">ROUGE-L {(metrics.rouge * 100).toFixed(1)}%</span>
                  )}
                  {metrics.recall != null && (
                    <span className="metric-tag">Recall {(metrics.recall * 100).toFixed(1)}%</span>
                  )}
                  {metrics.mrr != null && (
                    <span className="metric-tag">MRR {metrics.mrr.toFixed(2)}</span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AnswerDisplay;