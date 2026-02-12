import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import NewPrompt from "../../components/NewPrompt/NewPrompt";
import { chatAPI, messageAPI, healthCheck } from "../../services/api";

function ChatPage() {
  const { id } = useParams(); // chat id from route
  const navigate = useNavigate();
  const { token, loading: authLoading } = useAuth();

  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const endref = useRef(null);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !token) {
      navigate('/login');
    }
  }, [token, authLoading, navigate]);

  /* ---------------- Backend Health Check ---------------- */
  useEffect(() => {
    if (!token) return;

    const initBackend = async () => {
      try {
        await healthCheck();
        setBackendReady(true);
      } catch (err) {
        setBackendReady(false);
        setBackendError(err.message);
      }
    };
    initBackend();
  }, [token]);

  /* ---------------- Chat Init / Load ---------------- */
  useEffect(() => {
    if (!backendReady || !token || !id) return;

    const initChat = async () => {
      try {
        // Load messages for the chat
        const data = await messageAPI.getMessages(id);
        setMessages(data);
      } catch (error) {
        console.error('Error loading messages:', error);
        // If there's an error loading messages, initialize with empty messages
        setMessages([]);
      }
    };

    initChat();
  }, [backendReady, id, token, navigate]);

  /* ---------------- Auto-scroll ---------------- */
  useEffect(() => {
    endref.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* ---------------- Send Message ---------------- */
  const handleNewMessage = async (text) => {
    if (!text.trim() || !id || !token) return;

    try {
      setIsLoading(true);

      // Show user message immediately
      const tempMessageId = Date.now();
      setMessages((prev) => [
        ...prev,
        { content: text, role: "user", id: tempMessageId },
      ]);

      const response = await messageAPI.sendMessage(id, { content: text });

      setMessages((prev) => {
        // Remove the optimistic message and add both user and assistant messages
        const updatedMessages = prev.filter(msg => msg.id !== tempMessageId);
        return [...updatedMessages, 
          { content: text, role: "user", id: response.user_message_id || tempMessageId }, 
          response
        ];
      });
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          content: `Error: ${err.message}`,
          role: "assistant",
          error: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  /* ---------------- UI ---------------- */
  return (
    <div className="h-full flex flex-col items-center relative pb-24">
      {backendError && (
        <div className="w-full bg-red-900 text-white p-4 text-center">
          Backend Error: {backendError}
        </div>
      )}

      <div className="flex-1 overflow-scroll w-full flex justify-center">
        <div className="w-[50%] flex flex-col gap-5">
          {messages.map((msg, idx) => (
            <div
              key={msg.id || idx}
              className={`message ${msg.role === "user" ? "user" : ""}`}
            >
              {msg.role === "assistant" ? (
                <div dangerouslySetInnerHTML={{ __html: msg.content }} />
              ) : (
                msg.content
              )}
            </div>
          ))}

          {isLoading && (
            <div className="message">
              <div className="loading-dots">
                Thinking<span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
          <div ref={endref} />
        </div>
      </div>

      <div className="fixed bottom-0 w-full flex justify-center pb-6">
        <NewPrompt onSubmit={handleNewMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}

export default ChatPage;
