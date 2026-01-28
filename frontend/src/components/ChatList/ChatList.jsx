import { Link } from "react-router-dom";
import "./ChatList.css";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "../../context/AuthContext";
import { chatAPI } from "../../services/api";

const ChatList = () => {
  const { token } = useAuth();

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

  return (
    <div className="chatList">
      <span className="title">DASHBOARD</span>

      <Link to="/dashboard">➕ Create a new Chat</Link>
      <Link to="/">Explore MedGPT</Link>
      <Link to="/">Contact</Link>

      <hr />

      <span className="title">RECENT CHATS</span>

      <div className="list">
        {isPending && <span>Loading...</span>}

        {error && <span>Something went wrong!</span>}

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
