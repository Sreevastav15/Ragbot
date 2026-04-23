// src/components/Sidebar.jsx
import { useState, useEffect } from "react";
import { fileHistory } from "../api/chatHistory";
import { deleteChat } from "../api/delete";
import { toast } from "react-hot-toast";
import "./Sidebar.css";

function Sidebar({ isOpen, onClose, onNewChat, onSelectChat, refresh }) {
  const [fileList, setFileList] = useState([]);
  const [loading, setLoading] = useState(false);

  const getFiles = async () => {
    setLoading(true);
    try {
      const data = await fileHistory();
      setFileList(data);
    } catch (err) {
      console.error("Error fetching chat history", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) getFiles();
  }, [isOpen, refresh]);

  const handleDelete = async (doc_id, e) => {
    e.stopPropagation();
    const tid = toast.loading("Deleting…");
    try {
      await deleteChat(doc_id);
      setFileList((prev) => prev.filter((f) => f.doc_id !== doc_id));
      toast.success("Deleted", { id: tid });
      onNewChat();
    } catch {
      toast.error("Delete failed", { id: tid });
    }
  };

  const formatSize = (bytes) => {
    if (!bytes) return "";
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  return (
    <>
      {/* Backdrop */}
      {isOpen && <div className="sidebar-backdrop" onClick={onClose} />}

      <aside className={`sidebar ${isOpen ? "open" : ""}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <span className="sidebar-logo">◈</span>
            <span>RAGBot Pro</span>
          </div>
          <button className="sidebar-close-btn" onClick={onClose}>✕</button>
        </div>

        <button className="new-chat-btn" onClick={() => { onNewChat(); onClose(); }}>
          <span>＋</span> New Chat
        </button>

        <div className="sidebar-section">
          <p className="sidebar-section-label">Recent Chats</p>

          {loading && (
            <div className="sidebar-loading">
              <span className="sidebar-spinner" />
            </div>
          )}

          {!loading && fileList.length === 0 && (
            <p className="sidebar-empty">No chats yet</p>
          )}

          <ul className="sidebar-list">
            {fileList.map((file) => (
              <li key={file.doc_id} className="sidebar-item">
                <button
                  className="sidebar-item-btn"
                  onClick={() => { onSelectChat(file.doc_id); onClose(); }}
                >
                  <span className="item-icon">📄</span>
                  <span className="item-info">
                    <span className="item-name">{file.filename}</span>
                    <span className="item-meta">
                      {file.chunk_count ? `${file.chunk_count} chunks` : ""}
                      {file.file_size_bytes ? ` · ${formatSize(file.file_size_bytes)}` : ""}
                    </span>
                  </span>
                </button>
                <button
                  className="sidebar-delete"
                  onClick={(e) => handleDelete(file.doc_id, e)}
                  title="Delete"
                >
                  🗑
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="sidebar-footer">
          <p className="sidebar-footer-text">RAGBot Pro · 2025</p>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;