// src/pages/Home.jsx
import { useState, useRef, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import AnswerDisplay from "../components/AnswerDisplay";
import DocumentBadges from "../components/DocumentBadges";
import { uploadPDF, uploadMultiple } from "../api/uploadApi";
import { askQuestion } from "../api/qaApi";
import { fullChatHistory } from "../api/chatHistory";
import toast from "react-hot-toast";
import "./Home.css";

function Home() {
  // ── State ────────────────────────────────────────────────────────────────
  const [activeDocs,    setActiveDocs]    = useState([]);
  const [conversation,  setConversation]  = useState([]);
  const [loading,       setLoading]       = useState(false);
  const [refresh,       setRefresh]       = useState(false);
  const [sidebarOpen,   setSidebarOpen]   = useState(false);

  const uploadRef      = useRef(null);
  const messagesEndRef = useRef(null);

  // ── Scroll to bottom on new message ──────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation]);

  // ── File upload ───────────────────────────────────────────────────────────
  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;
    setLoading(true);
    const tid = toast.loading(
      `Uploading ${files.length > 1 ? files.length + " files" : files[0].name}…`
    );
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

  const handleFileChange  = (e) => handleUpload(e.target.files);
  const handleUploadClick = () => uploadRef.current?.click();
  const removeDoc = (docId) => setActiveDocs((prev) => prev.filter((d) => d.id !== docId));

  // ── Ask question (non-streaming) ──────────────────────────────────────────
  const handleAsk = async (question) => {
    if (activeDocs.length === 0) {
      toast.error("Upload at least one document first.");
      return;
    }

    const docIds = activeDocs.map((d) => d.id);

    // Add a placeholder — empty answer triggers thinking-dots in AnswerDisplay
    setConversation((prev) => [
      ...prev,
      { question, answer: "", sources: [], rewrittenQuery: null, responseTimeMs: null, isLoading: true },
    ]);
    setLoading(true);

    try {
      const data = await askQuestion(question, docIds);

      // Swap placeholder with the real answer
      setConversation((prev) =>
        prev.map((msg, i) =>
          i === prev.length - 1
            ? {
                question,
                answer:         data.answer          ?? "",
                sources:        data.sources         ?? [],
                rewrittenQuery: data.rewritten_query ?? null,
                responseTimeMs: data.response_time_ms ?? null,
                isLoading:      false,
              }
            : msg
        )
      );
    } catch (err) {
      console.error(err);
      toast.error("Failed to get answer.");
      setConversation((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // ── Load a past chat ──────────────────────────────────────────────────────
  const handleSelectChat = async (doc_id) => {
    const tid = toast.loading("Loading chat…");
    try {
      const data = await fullChatHistory(doc_id);
      setActiveDocs([{ id: doc_id, filename: data.filename, chunkCount: null }]);
      setConversation(
        data.conversation.map((c) => ({
          question:       c.question,
          answer:         c.answer,
          sources:        c.sources         ?? [],
          responseTimeMs: c.response_time_ms ?? null,
          rewrittenQuery: null,
          isLoading:      false,
        }))
      );
      toast.success(`Loaded "${data.filename}"`, { id: tid });
    } catch {
      toast.error("Failed to load chat", { id: tid });
    }
  };

  const handleNewChat = () => { setActiveDocs([]); setConversation([]); };

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen((p) => !p)} aria-label="Toggle sidebar">
            <span className="hamburger" />
          </button>
          <div className="logo-group">
            <span className="logo-icon">◈</span>
            <h1 className="logo-text">RAGBot</h1>
            <span className="logo-badge">Pro</span>
          </div>
        </div>
        <div className="header-right">
          <button className="upload-btn-header" onClick={handleUploadClick}>+ Upload</button>
        </div>
      </header>

      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        refresh={refresh}
      />

      {/* Main */}
      <main className={`app-main ${sidebarOpen ? "shifted" : ""}`}>

        {activeDocs.length > 0 && (
          <DocumentBadges docs={activeDocs} onRemove={removeDoc} />
        )}

        {conversation.length === 0 && activeDocs.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">◈</div>
            <h2>Chat with your documents</h2>
            <p>Upload one or more PDF, DOCX, or TXT files to get started.</p>
            <button className="empty-upload-btn" onClick={handleUploadClick}>Upload document</button>
          </div>
        )}

        {conversation.length === 0 && activeDocs.length > 0 && (
          <div className="empty-state">
            <div className="empty-icon" style={{ fontSize: "2rem" }}>💬</div>
            <h2>Ask anything about your document{activeDocs.length > 1 ? "s" : ""}</h2>
            <p>Your {activeDocs.length} document{activeDocs.length > 1 ? "s are" : " is"} ready.</p>
          </div>
        )}

        <div className="conversation">
          {conversation.map((msg, idx) => (
            <AnswerDisplay
              key={idx}
              question={msg.question}
              answer={msg.answer}
              sources={msg.sources}
              responseTimeMs={msg.responseTimeMs}
              rewrittenQuery={msg.rewrittenQuery}
              isLoading={msg.isLoading}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Chatbox */}
      <div className={`chatbox-dock ${sidebarOpen ? "shifted" : ""}`}>
        <ChatBox onAsk={handleAsk} loading={loading} docsLoaded={activeDocs.length > 0} />
      </div>

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