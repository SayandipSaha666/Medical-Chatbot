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
        
        <div className="text-center text-white">
          <p>Select a chat from the sidebar or create a new one</p>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;