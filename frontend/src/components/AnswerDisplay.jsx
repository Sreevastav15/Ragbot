// src/components/AnswerDisplay.jsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./AnswerDisplay.css";

/**
 * Renders one question/answer pair.
 *
 * Props
 * ─────
 * question       string   — the user's question
 * answer         string   — the LLM's answer (empty while loading)
 * sources        array    — [{ filename, page }]
 * responseTimeMs number   — backend latency in ms
 * rewrittenQuery string   — the rewritten retrieval query (optional)
 * isLoading      bool     — true while awaiting the non-streaming response
 */
function AnswerDisplay({
  question,
  answer,
  sources = [],
  responseTimeMs = null,
  rewrittenQuery = null,
  isLoading = false,
}) {
  if (!question) return null;

  const uniqueSources = Array.from(
    new Map(sources.map((s) => [`${s.filename}:${s.page}`, s])).values()
  );

  const showFooter = !isLoading && (answer || uniqueSources.length > 0);

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

          {/* Answer bubble */}
          <div className="bot-bubble">
            {isLoading || !answer ? (
              /* Thinking dots while waiting for response */
              <div className="thinking-dots">
                <span /><span /><span />
              </div>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
            )}
          </div>

          {/* Footer: timing + sources */}
          {showFooter && (
            <div className="answer-footer">
              {responseTimeMs != null && (
                <span className="timing-badge">⚡ {responseTimeMs}ms</span>
              )}

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
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

export default AnswerDisplay;