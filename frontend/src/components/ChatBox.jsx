// src/components/ChatBox.jsx
import { useState, useRef, useEffect } from "react";
import "./ChatBox.css";

function ChatBox({ onAsk, loading, docsLoaded }) {
  const [input, setInput] = useState("");
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
  }, [input]);

  const handleAsk = () => {
    const q = input.trim();
    if (!q || loading) return;
    onAsk(q);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className={`chatbox-wrapper ${!docsLoaded ? "disabled" : ""}`}>
      <textarea
        ref={textareaRef}
        rows={1}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKey}
        placeholder={docsLoaded ? "Ask a question… (Shift+Enter for newline)" : "Upload a document to start chatting"}
        disabled={!docsLoaded || loading}
        className="chatbox-textarea"
      />
      <button
        onClick={handleAsk}
        disabled={!input.trim() || loading || !docsLoaded}
        className="chatbox-send"
        aria-label="Send"
      >
        {loading ? (
          <span className="send-spinner" />
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        )}
      </button>
    </div>
  );
}

export default ChatBox;