import { Link, useNavigate } from "react-router-dom";
import "./ChatList.css";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "../../context/AuthContext";
import { chatAPI } from "../../services/api";
import { useState } from "react";

const ChatList = () => {
  const { token } = useAuth();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [newChatTitle, setNewChatTitle] = useState("");

  const { isPending, error, data } = useQuery({
    queryKey: ["userChats"],
    queryFn: async () => {
      if (!token) {
        throw new Error("Not authenticated");
      }

      return chatAPI.getChats();
    },
    enabled: !!token, // Only run query if token exists
  });

  const createChatMutation = useMutation({
    mutationFn: (title) => {
      return chatAPI.createChat({ title: title || "New Chat" });
    },
    onSuccess: (newChat) => {
      // Invalidate and refetch the user chats query
      queryClient.invalidateQueries({ queryKey: ["userChats"] });
      // Navigate to the new chat
      navigate(`/dashboard/chats/${newChat.id}`);
      // Reset the input field
      setNewChatTitle("");
    },
    onError: (error) => {
      console.error("Error creating chat:", error);
    }
  });

  const handleCreateChat = (e) => {
    e.preventDefault();
    if (newChatTitle.trim() || "New Chat".trim()) {
      createChatMutation.mutate(newChatTitle.trim() || "New Chat");
    }
  };

  return (
    <div className="chatList">
      <span className="title">DASHBOARD</span>

      <form onSubmit={handleCreateChat} className="create-chat-form">
        <input
          type="text"
          value={newChatTitle}
          onChange={(e) => setNewChatTitle(e.target.value)}
          placeholder="Enter chat title..."
          className="chat-title-input"
        />
        <button type="submit" disabled={createChatMutation.isPending} className="create-chat-btn">
          {createChatMutation.isPending ? "Creating..." : "➕"}
        </button>
      </form>

      <Link to="/">Explore MedGPT</Link>
      <Link to="/">Contact</Link>

      <hr />

      <span className="title">RECENT CHATS</span>

      <div className="list">
        {isPending && <span>Loading...</span>}

        {error && <span>Error: {error.message}</span>}

        {!isPending && !error && data?.length === 0 && (
          <span>No chats yet</span>
        )}

        {data?.map((chat) => (
          <Link
            to={`/dashboard/chats/${chat.id}`}
            key={chat.id}
            className="chatItem"
          >
            {chat.title || "Untitled Chat"}
          </Link>
        ))}
      </div>

      <hr />

      <div className="upgrade">
        <img src="/logo.png" alt="logo" />
        <div className="texts">
          <span>Upgrade to MedGPT Pro</span>
          <span>Get unlimited access to all features</span>
        </div>
      </div>
    </div>
  );
};

export default ChatList;
