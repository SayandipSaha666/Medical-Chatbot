import React, { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { useAuth } from "@clerk/clerk-react";
import NewPrompt from "../../components/NewPrompt/NewPrompt";
import { sendMessage, checkBackendHealth } from "../../services/chatService";

function ChatPage() {
  const { id: chatId } = useParams(); // /dashboard/chats/:id
  const { getToken } = useAuth();

  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const endref = useRef(null);

  useEffect(() => {
    const init = async () => {
      try {
        await checkBackendHealth();
        setBackendReady(true);
      } catch (err) {
        setBackendError(err.message);
      }
    };
    init();
  }, []);

  useEffect(() => {
    endref.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleNewMessage = async (text) => {
    if (!text.trim()) return;

    if (!backendReady) {
      setMessages((prev) => [
        ...prev,
        { content: "Backend not ready", role: "assistant", error: true },
      ]);
      return;
    }

    try {
      setIsLoading(true);

      // 1️⃣ Show user message immediately
      setMessages((prev) => [
        ...prev,
        { content: text, role: "user" },
      ]);

      const token = await getToken({ template: "backend" });

      // 2️⃣ Send to backend
      const assistantMessage = await sendMessage({
        chatId,
        message: text,
        token,
      });

      // 3️⃣ Append assistant response
      setMessages((prev) => [...prev, assistantMessage]);

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
              key={idx}
              className={`message ${msg.role === "user" ? "user" : ""}`}
            >
              {msg.role === "assistant" ? (
                <div
                  dangerouslySetInnerHTML={{ __html: msg.content }}
                />
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
