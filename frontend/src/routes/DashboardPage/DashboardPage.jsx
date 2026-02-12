import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { chatAPI, messageAPI } from "../../services/api";

function DashboardPage() {
  const navigate = useNavigate();
  const { token, loading: authLoading } = useAuth();
  const [chats, setChats] = useState([]);

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


  /* ---------------- Create new chat and navigate to it ---------------- */
  const createChat = async () => {
    if (!token) return;

    try {
      const chat = await chatAPI.createChat({ title: "New Chat" });
      setChats((prev) => [chat, ...prev]);
      
      // Navigate directly to the ChatPage for the new chat
      navigate(`/dashboard/chats/${chat.id}`);
    } catch (error) {
      console.error('Error creating chat:', error);
    }
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
        
        {/* Chat List */}
        <div className="w-full max-w-md">
          <h2 className="text-xl text-white mb-4">Your Chats</h2>
          {chats.length === 0 ? (
            <p className="text-gray-400">No chats yet. Start a new conversation!</p>
          ) : (
            <ul className="space-y-2">
              {chats.map((chat) => (
                <li 
                  key={chat.id}
                  className="p-3 bg-gray-700 rounded-lg text-white cursor-pointer hover:bg-gray-600"
                  onClick={() => navigate(`/dashboard/chats/${chat.id}`)}
                >
                  {chat.title}
                </li>
              ))}
            </ul>
          )}
          
          <button
            onClick={createChat}
            className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition duration-200"
          >
            + New Chat
          </button>
        </div>
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