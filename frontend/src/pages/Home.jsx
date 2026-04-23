// src/pages/Home.jsx
import { useState, useRef, useEffect, useCallback } from "react";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import AnswerDisplay from "../components/AnswerDisplay";
import DocumentBadges from "../components/DocumentBadges";
import MetricsPanel from "../components/MetricsPanel";
import { uploadPDF, uploadMultiple } from "../api/uploadApi";
import { askQuestionStream } from "../api/qaApi";
import { fullChatHistory } from "../api/chatHistory";
import toast from "react-hot-toast";
import "./Home.css";

function Home() {
  // ── State ────────────────────────────────────────────────────────────────
  const [activeDocs, setActiveDocs] = useState([]); // [{ id, filename, chunkCount }]
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [lastMetrics, setLastMetrics] = useState(null);

  const uploadRef = useRef(null);
  const multiUploadRef = useRef(null);
  const messagesEndRef = useRef(null);

  // ── Scroll to bottom on new message ──────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation]);

  // ── Single file upload ────────────────────────────────────────────────────
  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;

    setLoading(true);
    const tid = toast.loading(`Uploading ${files.length > 1 ? files.length + " files" : files[0].name}…`);

    try {
      if (files.length === 1) {
        const data = await uploadPDF(files[0]);
        setActiveDocs((prev) => [
          ...prev,
          { id: data.document_id, filename: data.filename, chunkCount: data.chunk_count },
        ]);
        toast.success(`"${data.filename}" ready`, { id: tid });
      } else {
        const data = await uploadMultiple(Array.from(files));
        const newDocs = data.uploaded
          .filter((d) => !d.error)
          .map((d) => ({ id: d.document_id, filename: d.filename, chunkCount: d.chunk_count }));
        setActiveDocs((prev) => [...prev, ...newDocs]);
        toast.success(`${newDocs.length} document(s) ready`, { id: tid });
      }
      setRefresh((p) => !p);
    } catch (err) {
      console.error(err);
      toast.error("Upload failed", { id: tid });
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => handleUpload(e.target.files);
  const handleUploadClick = () => uploadRef.current?.click();

  // ── Remove an active doc ──────────────────────────────────────────────────
  const removeDoc = (docId) => {
    setActiveDocs((prev) => prev.filter((d) => d.id !== docId));
  };

  // ── Ask question with streaming ───────────────────────────────────────────
  const handleAsk = async (question) => {
    if (activeDocs.length === 0) {
      toast.error("Upload at least one document first.");
      return;
    }

    const docIds = activeDocs.map((d) => d.id);

    // Optimistic add
    setConversation((prev) => [
      ...prev,
      { question, answer: "", streaming: true, sources: [], metrics: null, responseTimeMs: null },
    ]);

    setLoading(true);
    let metaReceived = null;
    const startTime = Date.now();

    try {
      await askQuestionStream(
        question,
        docIds,
        // onToken
        (token) => {
          setConversation((prev) =>
            prev.map((msg, i) =>
              i === prev.length - 1
                ? { ...msg, answer: msg.answer + token }
                : msg
            )
          );
        },
        // onMeta
        (meta) => {
          metaReceived = meta;
          setConversation((prev) =>
            prev.map((msg, i) =>
              i === prev.length - 1
                ? { ...msg, sources: meta.sources || [], rewrittenQuery: meta.rewritten_query }
                : msg
            )
          );
        },
        // onDone
        (done) => {
          const elapsed = done.response_time_ms || (Date.now() - startTime);
          setConversation((prev) =>
            prev.map((msg, i) =>
              i === prev.length - 1
                ? { ...msg, streaming: false, responseTimeMs: elapsed }
                : msg
            )
          );
          setLastMetrics({ responseTimeMs: elapsed, sources: metaReceived?.sources || [] });
        }
      );
    } catch (err) {
      toast.error("Failed to get answer.");
      setConversation((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // ── Load a past chat from sidebar ─────────────────────────────────────────
  const handleSelectChat = async (doc_id) => {
    toast.loading("Loading chat…");
    try {
      const data = await fullChatHistory(doc_id);
      setActiveDocs([{ id: doc_id, filename: data.filename, chunkCount: null }]);
      setConversation(
        data.conversation.map((c) => ({
          question: c.question,
          answer: c.answer,
          sources: c.sources || [],
          responseTimeMs: c.response_time_ms,
          metrics: c.metrics,
          streaming: false,
        }))
      );
      toast.dismiss();
      toast.success(`Loaded "${data.filename}"`);
    } catch (err) {
      toast.dismiss();
      toast.error("Failed to load chat");
    }
  };

  const handleNewChat = () => {
    setActiveDocs([]);
    setConversation([]);
    setLastMetrics(null);
  };

  return (
    <div className="app-shell">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-left">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((p) => !p)}
            aria-label="Toggle sidebar"
          >
            <span className="hamburger" />
          </button>
          <div className="logo-group">
            <span className="logo-icon">◈</span>
            <h1 className="logo-text">RAGBot</h1>
            <span className="logo-badge">Pro</span>
          </div>
        </div>

        <div className="header-right">
          {lastMetrics && (
            <button
              className="metrics-pill"
              onClick={() => setMetricsOpen((p) => !p)}
              title="Toggle metrics panel"
            >
              ⚡ {lastMetrics.responseTimeMs}ms
            </button>
          )}
          <button className="upload-btn-header" onClick={handleUploadClick}>
            + Upload
          </button>
        </div>
      </header>

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        refresh={refresh}
      />

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className={`app-main ${sidebarOpen ? "shifted" : ""}`}>

        {/* Metrics panel */}
        {metricsOpen && lastMetrics && (
          <MetricsPanel metrics={lastMetrics} onClose={() => setMetricsOpen(false)} />
        )}

        {/* Active document badges */}
        {activeDocs.length > 0 && (
          <DocumentBadges docs={activeDocs} onRemove={removeDoc} />
        )}

        {/* Empty state */}
        {conversation.length === 0 && activeDocs.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">◈</div>
            <h2>Chat with your documents</h2>
            <p>Upload one or more PDF, DOCX, or TXT files to get started.</p>
            <button className="empty-upload-btn" onClick={handleUploadClick}>
              Upload document
            </button>
          </div>
        )}

        {/* Empty state – docs loaded but no messages yet */}
        {conversation.length === 0 && activeDocs.length > 0 && (
          <div className="empty-state">
            <div className="empty-icon" style={{ fontSize: "2rem" }}>💬</div>
            <h2>Ask anything about your document{activeDocs.length > 1 ? "s" : ""}</h2>
            <p>Your {activeDocs.length} document{activeDocs.length > 1 ? "s are" : " is"} ready.</p>
          </div>
        )}

        {/* Conversation */}
        <div className="conversation">
          {conversation.map((msg, idx) => (
            <AnswerDisplay
              key={idx}
              question={msg.question}
              answer={msg.answer}
              sources={msg.sources}
              metrics={msg.metrics}
              responseTimeMs={msg.responseTimeMs}
              rewrittenQuery={msg.rewrittenQuery}
              isStreaming={msg.streaming}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* ── Bottom chatbox ────────────────────────────────────────────────── */}
      <div className={`chatbox-dock ${sidebarOpen ? "shifted" : ""}`}>
        <ChatBox onAsk={handleAsk} loading={loading} docsLoaded={activeDocs.length > 0} />
      </div>

      {/* Hidden file inputs */}
      <input
        type="file"
        accept=".pdf,.docx,.txt"
        ref={uploadRef}
        onChange={handleFileChange}
        multiple
        className="hidden-input"
      />
    </div>
  );
}

export default Home;