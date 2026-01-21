import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import fetchWithAuth from "../../utils/fetchWithAuth";

function DashboardPage() {
  const navigate = useNavigate();

  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [loading, setLoading] = useState(false);

  /* ---------------- Fetch all chats ---------------- */
  useEffect(() => {
    fetchWithAuth("/api/chats")
      .then((res) => res.json())
      .then((data) => setChats(data))
      .catch(console.error);
  }, []);

  /* ---------------- Load messages when chat changes ---------------- */
  useEffect(() => {
    if (!chatId) return;

    fetchWithAuth(`/api/chats/${chatId}/messages`)
      .then((res) => res.json())
      .then((data) => setMessages(data))
      .catch(console.error);
  }, [chatId]);

  /* ---------------- Create chat if needed ---------------- */
  const createChat = async () => {
    const res = await fetchWithAuth("/api/chats", {
      method: "POST",
      body: JSON.stringify({ title: "New Chat" }),
    });

    const chat = await res.json();
    setChats((prev) => [chat, ...prev]);
    setChatId(chat.id);

    // Optional: navigate to chat page
    navigate(`/dashboard/chats/${chat.id}`);
    return chat.id;
  };

  /* ---------------- Send message ---------------- */
  const handleSend = async (e) => {
    e.preventDefault();
    if (!newMessage.trim()) return;

    setLoading(true);

    let activeChatId = chatId;
    if (!activeChatId) {
      activeChatId = await createChat();
    }

    // Optimistic user message
    setMessages((prev) => [
      ...prev,
      { role: "user", content: newMessage },
    ]);

    const res = await fetchWithAuth(
      `/api/chats/${activeChatId}/messages`,
      {
        method: "POST",
        body: JSON.stringify({ content: newMessage }),
      }
    );

    const data = await res.json();

    setMessages((prev) => [...prev, data]);
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
            key={idx}
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