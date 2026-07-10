import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

// ── Config ─────────────────────────────────────────────────────────────────────
// P3: Use env variable instead of hardcoded URL.
// Set VITE_API_URL in frontend/.env for each environment.
const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const STORAGE_KEY = "chat_context";

const generateId = () =>
  "msg_" + Date.now() + "_" + Math.random().toString(36).slice(2);

const getTimestamp = () => new Date().toISOString();

function App() {
  const [query, setQuery] = useState("");

  const [messages, setMessages] = useState(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return parsed.context || [];
      }
      return [];
    } catch {
      return [];
    }
  });

  const [loading, setLoading] = useState(false);

  // ── P10: Upload state ───────────────────────────────────────────────────────
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // {type: "success"|"error", message}
  const [showUpload, setShowUpload] = useState(false);
  const fileInputRef = useRef(null);

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Save to sessionStorage
  useEffect(() => {
    const limitedMessages = messages.slice(-50);
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ context: limitedMessages })
    );
  }, [messages]);

  const streamResponse = async (fullText, callback) => {
    let current = "";
    for (let i = 0; i < fullText.length; i++) {
      current += fullText[i];
      callback(current);
      await new Promise((res) => setTimeout(res, 10));
    }
  };

  const handleSearch = async () => {
    if (!query.trim() || loading) return;

    setLoading(true);

    const userMessage = {
      id: generateId(),
      role: "user",
      content: query,
      timestamp: getTimestamp(),
    };

    // P2: Build the payload from existing messages + user message ONLY.
    // The bot placeholder is added to local state for rendering but NOT sent to the API.
    const apiMessages = [...messages, userMessage];

    const botMessage = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: getTimestamp(),
    };

    setMessages([...apiMessages, botMessage]);
    setQuery("");

    try {
      // P3: Use API_BASE instead of hardcoded localhost URL
      const response = await axios.post(`${API_BASE}/chat`, {
        messages: apiMessages, // P2: no empty bot placeholder
      });

      const botReply = response.data.reply || "No response from server.";

      await streamResponse(botReply, (partialText) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: partialText,
          };
          return updated;
        });
      });
    } catch (error) {
      const detail =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        error.message ||
        "Unknown error";

      console.error("Chat ERROR:", detail);

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: `⚠️ Error: ${detail}`,
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    sessionStorage.removeItem(STORAGE_KEY);
  };

  // ── P10: PDF Upload handler ─────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!uploadFile || uploading) return;

    if (uploadFile.type !== "application/pdf") {
      setUploadStatus({ type: "error", message: "Only PDF files are accepted." });
      return;
    }

    setUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", uploadFile);

    try {
      const response = await axios.post(`${API_BASE}/ingest`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { chunks_indexed, batches_failed } = response.data;
      setUploadStatus({
        type: "success",
        message: `✅ Ingested "${uploadFile.name}" — ${chunks_indexed} chunks indexed${
          batches_failed > 0 ? `, ${batches_failed} batch(es) failed` : ""
        }.`,
      });
      setUploadFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      const detail =
        error.response?.data?.detail || error.message || "Upload failed.";
      setUploadStatus({ type: "error", message: `❌ ${detail}` });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="app-container">

      {/* HEADER */}
      <div className="header">
        <h1>Synapse</h1>
        <div className="header-actions">
          <button
            className="btn-secondary"
            onClick={() => {
              setShowUpload((v) => !v);
              setUploadStatus(null);
            }}
          >
            {showUpload ? "← Chat" : "📄 Upload PDF"}
          </button>
          <button className="btn-secondary" onClick={clearChat}>
            Clear Chat
          </button>
        </div>
      </div>

      {/* UPLOAD PANEL (P10) */}
      {showUpload && (
        <div className="upload-panel">
          <h2>Upload a PDF to the Knowledge Base</h2>
          <p className="upload-hint">
            The document will be chunked, embedded, and indexed into Qdrant so
            Synapse can answer questions about it.
          </p>

          <div className="upload-controls">
            <input
              id="pdf-file-input"
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={(e) => {
                setUploadFile(e.target.files[0] || null);
                setUploadStatus(null);
              }}
              className="file-input"
            />
            <button
              id="upload-submit-btn"
              onClick={handleUpload}
              disabled={!uploadFile || uploading}
              className="btn-primary"
            >
              {uploading ? "Uploading…" : "Ingest PDF"}
            </button>
          </div>

          {uploadFile && !uploadStatus && (
            <p className="upload-hint">Selected: <strong>{uploadFile.name}</strong></p>
          )}

          {uploadStatus && (
            <p className={`upload-status ${uploadStatus.type}`}>
              {uploadStatus.message}
            </p>
          )}
        </div>
      )}

      {/* CHAT */}
      {!showUpload && (
        <div className="chat-container">
          {messages.length === 0 ? (
            <p className="empty">Start the conversation…</p>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`chat-bubble ${msg.role === "user" ? "user" : "bot"}`}
              >
                <pre>{msg.content}</pre>
              </div>
            ))
          )}
          <div ref={bottomRef}></div>
        </div>
      )}

      {/* INPUT */}
      {!showUpload && (
        <div className="input-section">
          <input
            id="chat-input"
            type="text"
            placeholder="Send a message…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && query.trim() && !loading) {
                handleSearch();
              }
            }}
          />

          <button id="chat-send-btn" onClick={handleSearch} disabled={loading}>
            {loading ? "Sending…" : "Send"}
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
