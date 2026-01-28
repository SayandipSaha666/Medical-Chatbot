// import React, { useState, useEffect, useRef } from "react";
// import { useParams } from "react-router-dom";
// import { useAuth } from "@clerk/clerk-react";
// import NewPrompt from "../../components/NewPrompt/NewPrompt";
// import { sendMessage, checkBackendHealth } from "../../services/chatService";

// function ChatPage() {
//   const { id: chatId } = useParams(); // /dashboard/chats/:id
//   const { getToken } = useAuth();

//   const [messages, setMessages] = useState([]);
//   const [isLoading, setIsLoading] = useState(false);
//   const [backendReady, setBackendReady] = useState(false);
//   const [backendError, setBackendError] = useState(null);
//   const endref = useRef(null);

//   useEffect(() => {
//     const init = async () => {
//       try {
//         await checkBackendHealth();
//         setBackendReady(true);
//       } catch (err) {
//         setBackendError(err.message);
//       }
//     };
//     init();
//   }, []);

//   useEffect(() => {
//     endref.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   const handleNewMessage = async (text) => {
//     if (!text.trim()) return;

//     if (!backendReady) {
//       setMessages((prev) => [
//         ...prev,
//         { content: "Backend not ready", role: "assistant", error: true },
//       ]);
//       return;
//     }

//     try {
//       setIsLoading(true);

//       // 1️⃣ Show user message immediately
//       setMessages((prev) => [
//         ...prev,
//         { content: text, role: "user" },
//       ]);

//       const token = await getToken({ template: "backend" });

//       // 2️⃣ Send to backend
//       const assistantMessage = await sendMessage({
//         chatId,
//         message: text,
//         token,
//       });

//       // 3️⃣ Append assistant response
//       setMessages((prev) => [...prev, assistantMessage]);

//     } catch (err) {
//       setMessages((prev) => [
//         ...prev,
//         {
//           content: `Error: ${err.message}`,
//           role: "assistant",
//           error: true,
//         },
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <div className="h-full flex flex-col items-center relative pb-24">
//       {backendError && (
//         <div className="w-full bg-red-900 text-white p-4 text-center">
//           Backend Error: {backendError}
//         </div>
//       )}

//       <div className="flex-1 overflow-scroll w-full flex justify-center">
//         <div className="w-[50%] flex flex-col gap-5">
//           {messages.map((msg, idx) => (
//             <div
//               key={idx}
//               className={`message ${msg.role === "user" ? "user" : ""}`}
//             >
//               {msg.role === "assistant" ? (
//                 <div
//                   dangerouslySetInnerHTML={{ __html: msg.content }}
//                 />
//               ) : (
//                 msg.content
//               )}
//             </div>
//           ))}

//           {isLoading && (
//             <div className="message">
//               <div className="loading-dots">
//                 Thinking<span>.</span><span>.</span><span>.</span>
//               </div>
//             </div>
//           )}
//           <div ref={endref} />
//         </div>
//       </div>

//       <div className="fixed bottom-0 w-full flex justify-center pb-6">
//         <NewPrompt onSubmit={handleNewMessage} isLoading={isLoading} />
//       </div>
//     </div>
//   );
// }

// export default ChatPage;

import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import NewPrompt from "../../components/NewPrompt/NewPrompt";
import { chatAPI, messageAPI, healthCheck } from "../../services/api";

function ChatPage() {
  const { id } = useParams(); // chat id from route
  const navigate = useNavigate();
  const { token, loading: authLoading } = useAuth();

  const [chatId, setChatId] = useState(id || null);
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
    if (!backendReady || !token || !chatId) return;

    const initChat = async () => {
      try {
        // Load messages for the chat
        const data = await messageAPI.getMessages(chatId);
        setMessages(data);
      } catch (error) {
        console.error('Error loading messages:', error);

        // If there's an error loading messages, create a new chat
        try {
          const newChat = await chatAPI.createChat({ title: "New Chat" });
          setChatId(newChat.id);
          navigate(`/dashboard/chats/${newChat.id}`, { replace: true });
          setMessages([]);
        } catch (createError) {
          console.error('Error creating new chat:', createError);
        }
      }
    };

    initChat();
  }, [backendReady, chatId, token, navigate]);

  /* ---------------- Auto-scroll ---------------- */
  useEffect(() => {
    endref.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* ---------------- Send Message ---------------- */
  const handleNewMessage = async (text) => {
    if (!text.trim() || !chatId || !token) return;

    try {
      setIsLoading(true);

      // Show user message immediately
      setMessages((prev) => [
        ...prev,
        { content: text, role: "user", id: Date.now() },
      ]);

      const assistantMessage = await messageAPI.sendMessage(chatId, { content: text });

      setMessages((prev) => {
        // Remove the optimistic message and add both user and assistant messages
        const updatedMessages = prev.filter(msg => msg.id !== Date.now());
        return [...updatedMessages, assistantMessage];
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
