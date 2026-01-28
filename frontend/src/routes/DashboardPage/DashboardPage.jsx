import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { chatAPI, messageAPI } from "../../services/api";

function DashboardPage() {
  const navigate = useNavigate();
  const { token, loading: authLoading } = useAuth();
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [loading, setLoading] = useState(false);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !token) {
      navigate('/login');
    }
  }, [token, authLoading, navigate]);

  /* ---------------- Fetch all chats ---------------- */
  useEffect(() => {
    if (!token) return;

    const fetchChats = async () => {
      try {
        const data = await chatAPI.getChats();
        setChats(data);
      } catch (error) {
        console.error('Error fetching chats:', error);
      }
    };

    fetchChats();
  }, [token]);

  /* ---------------- Load messages when chat changes ---------------- */
  useEffect(() => {
    if (!chatId || !token) return;

    const fetchMessages = async () => {
      try {
        const data = await messageAPI.getMessages(chatId);
        setMessages(data);
      } catch (error) {
        console.error('Error fetching messages:', error);
      }
    };

    fetchMessages();
  }, [chatId, token]);

  /* ---------------- Create chat if needed ---------------- */
  const createChat = async () => {
    if (!token) return;

    try {
      const chat = await chatAPI.createChat({ title: "New Chat" });
      setChats((prev) => [chat, ...prev]);
      setChatId(chat.id);

      // Navigate to the new chat
      navigate(`/dashboard/chats/${chat.id}`);
      return chat.id;
    } catch (error) {
      console.error('Error creating chat:', error);
      return null;
    }
  };

  /* ---------------- Send message ---------------- */
  const handleSend = async (e) => {
    e.preventDefault();
    if (!newMessage.trim() || !token) return;

    setLoading(true);

    let activeChatId = chatId;
    if (!activeChatId) {
      activeChatId = await createChat();
      if (!activeChatId) {
        setLoading(false);
        return;
      }
    }

    // Optimistic user message
    setMessages((prev) => [
      ...prev,
      { role: "user", content: newMessage, id: Date.now() } // Using timestamp as temporary ID
    ]);

    try {
      const data = await messageAPI.sendMessage(activeChatId, { content: newMessage });
      setMessages(prev => {
        // Remove the optimistic message and add both user and assistant messages
        const updatedMessages = prev.filter(msg => msg.id !== Date.now());
        return [...updatedMessages, data];
      });
    } catch (error) {
      console.error('Error sending message:', error);
      // Remove the optimistic message in case of error
      setMessages(prev => prev.filter(msg => msg.id !== Date.now()));
    }

    setNewMessage("");
    setLoading(false);
  };

  /* ---------------- UI ---------------- */
  return (
    <div className="h-full flex flex-col items-center">
      {/* Header */}
      <div className="flex-1 flex flex-col items-center justify-center w-[50%] gap-12">
        <div className="flex items-center gap-5 opacity-20">
          <img src="/logo.png" alt="MedGPT" className="w-40 rounded-full" />
          <h1 className="text-[128px] bg-linear-to-r from-[#217bfe] to-[#e55571] bg-clip-text text-transparent">
            MedGPT
          </h1>
        </div>
      </div>

      {/* Messages */}
      <div className="w-[50%] flex flex-col gap-4 mb-6 overflow-y-auto">
        {messages.map((msg, idx) => (
          <div
            key={msg.id || idx}
            className={`p-4 rounded-xl ${
              msg.role === "user"
                ? "bg-[#217bfe] self-end text-white"
                : "bg-[#2c2937] self-start text-[#ececec]"
            }`}
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
      </div>

      {/* Input */}
      <div className="mt-auto w-[50%] bg-[#2c2937] rounded-2xl flex">
        <form
          onSubmit={handleSend}
          className="w-full h-full flex items-center justify-between gap-5 mb-3"
        >
          <input
            type="text"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            placeholder="Ask me anything..."
            className="flex-1 p-5 bg-transparent text-[#ececec] border-none outline-none"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-[#605e68] rounded-full p-3 flex items-center justify-center mr-5"
          >
            <img src="/arrow.png" alt="submit" className="w-8" />
          </button>
        </form>
      </div>
    </div>
  );
}

export default DashboardPage;


// import React from 'react'
// const handleSubmit = () => {
  
// }
// function DashboardPage() {
//   return (
//     <div className="h-full flex flex-col items-center">
//       <div className="flex-1 flex flex-col items-center justify-center w-[50%] gap-12">
//         <div className="flex items-center gap-5 opacity-20">
//           <img src="/logo.png" alt="MedGPT" className='w-40 rounded-full'/>
//           <h1 className='text-[128px] bg-linear-to-r from-[#217bfe] to-[#e55571] bg-clip-text text-transparent'>MedGPT</h1>
//         </div>
//       </div>
//       <div className="mt-auto w-[50%] bg-[#2c2937] rounded-2xl flex">
//         <form onSubmit={handleSubmit} className='w-full h-full flex items-center justify-between gap-5 mb-3'>
//           <input type="text" name="text" placeholder="Ask me anything..." className='flex-1 p-5 bg-transparent text-[#ececec] border-none outline-none w-full'/>
//           <button className='bg-[#605e68] rounded-full border-none cursor-pointer p-3 flex items-center justify-center mr-5 '>
//             <img src="/arrow.png" alt="submit" className='w-8'/>
//           </button>
//         </form>
//       </div>
//     </div>
//   )
// }

// export default DashboardPage