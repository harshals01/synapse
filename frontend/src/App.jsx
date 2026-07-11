import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const API_KEY = import.meta.env.VITE_API_KEY || "";
const authHeaders = () => (API_KEY ? { "X-API-Key": API_KEY } : {});

const STORAGE_KEY = "chat_context";

const generateId = () =>
  "msg_" + Date.now() + "_" + Math.random().toString(36).slice(2);

const getTimestamp = () => new Date().toISOString();

const formatMessage = (text) => {
  if (!text) return "";
  const tokens = text.split("**");
  return tokens.map((token, i) => {
    if (i % 2 === 1) {
      return <strong key={i}>{token}</strong>;
    }
    return token;
  });
};

const formatTime = (isoString) => {
  if (!isoString) return "";
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
};

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

  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // {type: "success"|"error", message}
  const [showUpload, setShowUpload] = useState(false);
  const fileInputRef = useRef(null);

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
      const response = await axios.post(
        `${API_BASE}/chat`,
        { messages: apiMessages },
        { headers: { ...authHeaders() } },
      );

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
        headers: {
          "Content-Type": "multipart/form-data",
          ...authHeaders(),
        },
      });

      const { chunks_indexed, batches_failed } = response.data;
      setUploadStatus({
        type: "success",
        message: ` Ingested "${uploadFile.name}" — ${chunks_indexed} chunks indexed${batches_failed > 0 ? `, ${batches_failed} batch(es) failed` : ""
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
        <div className="brand">
          <svg className="brand-icon" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2z"/>
            <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"/>
          </svg>
          <h1>Synapse</h1>
        </div>
        <div className="header-actions">
          <button
            className="btn-secondary header-btn"
            onClick={() => {
              setShowUpload((v) => !v);
              setUploadStatus(null);
            }}
          >
            {showUpload ? (
              <>
                <svg className="btn-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="19" y1="12" x2="5" y2="12"></line>
                  <polyline points="12 19 5 12 12 5"></polyline>
                </svg>
                <span>Chat</span>
              </>
            ) : (
              <>
                <svg className="btn-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <span>Upload PDF</span>
              </>
            )}
          </button>
          <button className="btn-secondary header-btn" onClick={clearChat}>
            <svg className="btn-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              <line x1="10" y1="11" x2="10" y2="17"></line>
              <line x1="14" y1="11" x2="14" y2="17"></line>
            </svg>
            <span>Clear Chat</span>
          </button>
        </div>
      </div>

      {/* UPLOAD PANEL */}
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
            messages.map((msg) => {
              const isUser = msg.role === "user";
              return (
                <div key={msg.id} className={`chat-item ${isUser ? "user" : "bot"}`}>
                  <div className="chat-tag">{isUser ? "User" : "Synapse"}</div>
                  <div className={`chat-bubble ${isUser ? "user" : "bot"}`}>
                    <div className="msg-text">{formatMessage(msg.content)}</div>
                    <span className="bubble-timestamp">{formatTime(msg.timestamp)}</span>
                  </div>
                </div>
              );
            })
          )}
          <div ref={bottomRef}></div>
        </div>
      )}

      {/* INPUT */}
      {!showUpload && (
        <div className="input-section">
          <div className="input-wrapper">
            <button className="input-icon-btn attachment-btn" type="button" title="Attach file">
              <svg className="input-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
              </svg>
            </button>
            <input
              id="chat-input"
              type="text"
              placeholder="Ask anything about your PDF…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && query.trim() && !loading) {
                  handleSearch();
                }
              }}
            />
            <button
              id="chat-send-btn"
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              title="Send message"
            >
              {loading ? (
                <span className="spinner"></span>
              ) : (
                <svg className="send-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
