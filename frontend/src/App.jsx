import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

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
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Save to sessionStorage 
  useEffect(() => {
    const limitedMessages = messages.slice(-50);

    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        context: limitedMessages
      })
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
      timestamp: getTimestamp()
    };

    const botMessage = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: getTimestamp()
    };

    const newMessages = [...messages, userMessage, botMessage];
    setMessages(newMessages);

    try {
      const response = await axios.post("http://127.0.0.1:8000/chat", {
        messages: newMessages
      });

      const botReply = response.data.reply || "No response from server.";

      await streamResponse(botReply, (partialText) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: partialText
          };
          return updated;
        });
      });

      setQuery("");

    } catch (error) {
      console.error("ERROR:", error.response?.data || error.message);

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: "Error fetching response."
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

  return (
    <div className="app-container">

      {/* HEADER */}
      <div className="header">
        <h1>Synapse</h1>
        <button onClick={clearChat}>Clear Chat</button>
      </div>

      {/* CHAT */}
      <div className="chat-container">
        {messages.length === 0 ? (
          <p className="empty">Start the conversation...</p>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-bubble ${msg.role === "user" ? "user" : "bot"
                }`}
            >
              <pre>{msg.content}</pre>
            </div>
          ))
        )}
        <div ref={bottomRef}></div>
      </div>

      {/* INPUT */}
      <div className="input-section">
        <input
          type="text"
          placeholder="Send a message..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && query.trim() && !loading) {
              handleSearch();
            }
          }}
        />

        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default App;
